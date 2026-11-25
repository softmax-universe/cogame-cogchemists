## Cogchemists game server: implements the Coworld game contract.
##
## Endpoints:
##   GET /healthz                    - liveness
##   GET /client/global              - spectator page
##   GET /client/player              - player page (view-only; policies are prompts)
##   GET /client/replay              - replay page (replay mode)
##   GET /client/renderer.js         - shared stage renderer
##   GET /client/chrome.css          - shared chrome
##   GET /client/assets/<name>       - sprites and fonts
##   WS  /player?slot=N&token=T      - player protocol
##   WS  /global                     - spectator snapshots
##   WS  /replay                     - replay payload (replay mode)
##
## Player protocol (cogchemists.player.v2), all JSON text frames:
##   game -> player: {"type":"welcome","slot":N,"name":...,"rounds":N}
##                   {"type":"state",...} after every event, redacted to
##                   exactly what that seat may see
##                   {"type":"final","scores":[...],...}
##   player -> game: {"type":"prompt","prompt":"...","scripted":"assayer"}
##                   (max 4000 runes; scripted plays a built-in baseline for
##                   that seat: "assayer" / "1", or "quack")
##                   {"type":"register","control":"external"}
##                   {"type":"action","id":N,"move":"exact legal label"}
##   game -> external player: {"type":"observation","id":N,
##                   "observation":<acting seat's redacted state>}

import
  std/[json, locks, os, sets, strutils, tables, times, unicode],
  bitworld/runtime,
  curly,
  mummy,
  mummy/routers,
  llm,
  training,
  prompts,
  sim

const
  MaxPromptLen = 4000
  ReplayVersion = 1

type
  GameState = object
    trajectory: DecisionTrajectory
    config: GameConfig
    selection: PromptSelection
    sim: Sim
    prompts: seq[string]
    scripted: seq[ScriptKind]
    external: seq[bool]
    registered: seq[bool]
    decisionId: int
    pending: seq[bool]
    accepted: seq[bool]
    actions: seq[Action]
    lastRawInput: seq[string]
    lastRejection: seq[string]
    playerSockets: Table[int, WebSocket]
    socketSlots: Table[WebSocket, int]
    globalSockets: HashSet[WebSocket]
    started: bool
    finished: bool

var
  stateLock: Lock
  state: GameState
  gameServer: Server
  runtimeConfigGlobal: RuntimeConfig
  replayPayloadGlobal: string

initLock(stateLock)

proc clientDir(): string =
  let appDir = getAppDir()
  for candidate in [appDir / "client", appDir / ".." / "client", "client"]:
    if dirExists(candidate):
      return candidate
  "client"

proc dataDir(): string =
  let appDir = getAppDir()
  for candidate in [appDir / "data", appDir / ".." / "data", "data"]:
    if dirExists(candidate):
      return candidate
  "data"

proc policyNamesJson(gs: GameState): JsonNode =
  ## Seats play under anonymous table names; the policy names ride alongside
  ## for the SPECTATOR views only, which render them in place of the aliases.
  result = newJArray()
  for player in gs.config.players:
    result.add(%player.name)

proc snapshotJson(gs: GameState): JsonNode =
  var events = newJArray()
  for event in gs.sim.events:
    events.add(event.eventToJson())
  var connected = newJArray()
  for slot in 0 ..< gs.config.tokens.len:
    connected.add(%gs.playerSockets.hasKey(slot))
  result = gs.sim.tableStateJson()
  result["type"] = %"state"
  result["game"] = %"cogchemists"
  result["policyNames"] = gs.policyNamesJson()
  result["events"] = events
  result["started"] = %gs.started
  result["done"] = %gs.sim.done
  result["connected"] = connected

proc playerStateJson(gs: GameState, slot: int): JsonNode =
  ## Redacted to exactly the seat's own world: no rival's hand, private
  ## facts, notes, grid or chemistriesLeft, and never the chemistry.
  result = gs.sim.observationJson(slot)
  result["started"] = %gs.started

proc broadcastLocked(gs: GameState) =
  ## Callers hold stateLock. Spectators get the whole table (including the
  ## chemistry once the exhibition has run); players get the redacted frame.
  let payload = $gs.snapshotJson()
  for socket in gs.globalSockets:
    socket.send(payload)
  for slot, socket in gs.playerSockets:
    socket.send($gs.playerStateJson(slot))

proc writeArtifact(uri, data, contentType, methodEnv: string) =
  ## Writes a Coworld artifact, honoring the platform's PUT/POST method hint.
  if uri.len == 0:
    return
  let httpMethod = getEnv(methodEnv, "PUT").toUpperAscii()
  if uri.isHttpCogameUri() and httpMethod == "POST":
    let curl = newCurly()
    var headers: HttpHeaders
    headers["content-type"] = contentType
    let response = curl.post(uri, headers, data, 60)
    if response.code < 200 or response.code >= 300:
      raise newException(IOError,
        "artifact POST failed: " & $response.code)
  else:
    writeCogameUri(uri, data, contentType, methodEnv)

proc replayPayload(gs: GameState, results: JsonNode): string =
  var names = newJArray()
  for name in gs.sim.names:
    names.add(%name)
  var events = newJArray()
  for event in gs.sim.events:
    events.add(event.eventToJson())
  var chemistry = newJArray()
  for index in 0 ..< Ingredients:
    chemistry.add(%gs.sim.chemistry[index])
  $ %*{
    "protocol": "cogchemists.replay.v" & $ReplayVersion,
    "names": names,
    "policyNames": gs.policyNamesJson(),
    "config": {
      "rounds": gs.config.rounds,
      "seed": gs.config.seed,
      "talk": gs.config.talk,
      "artifacts": gs.config.artifacts,
      "sampled": true,
      "chemistry": chemistry
    },
    "events": events,
    "results": results
  }

proc statesFromEvents(config: GameConfig, events: seq[GameEvent]): JsonNode =
  ## One table-state object per event prefix, for scrubbing replays.
  result = newJArray()
  for frame in replayMatch(config, events):
    result.add(frame.tableStateJson())

proc finishEpisode(runtimeConfig: RuntimeConfig) =
  var results: JsonNode
  var replayData: string
  var grace = 20
  withLock stateLock:
    if state.finished:
      return
    state.finished = true
    grace = state.config.shutdownGraceSeconds
    results = state.sim.resultsJson()
    replayData = state.replayPayload(results)
    state.trajectory.finishNativeTrajectory(state.sim)

    ## Send final frames to players BEFORE writing artifacts: the hosted
    ## worker tears player pods down as soon as results.json exists, and
    ## writing first would race player log collection.
    ## Results carry POLICY names for the platform; the final frame goes to
    ## the player sockets, so hand them the table aliases instead.
    var aliasNames = newJArray()
    for name in state.sim.names:
      aliasNames.add(%name)
    var final = %*{
      "type": "final",
      "done": true,
      "scores": results["scores"],
      "reputation": results["reputation"],
      "coin": results["coin"],
      "names": aliasNames,
      "rounds": results["rounds"],
      "reason": results["reason"]
    }
    for slot, socket in state.playerSockets:
      final["slot"] = %slot
      socket.send($final)
    state.broadcastLocked()

  sleep(500)
  echo "cogchemists: writing results and replay"
  writeArtifact(
    runtimeConfig.resultsUri, $results, "application/json",
    "COGAME_RESULTS_METHOD"
  )
  writeArtifact(
    runtimeConfig.replayUri, replayData, "application/octet-stream",
    "COGAME_SAVE_REPLAY_METHOD"
  )
  ## Keep /healthz and /global answering for a bounded grace after the
  ## artifacts land: the certifier pings /global AFTER the player pods
  ## start, and a short episode would otherwise already be gone.
  echo "cogchemists: artifacts written; serving for ", grace,
    "s before shutdown"
  sleep(max(0, grace) * 1000)
  echo "cogchemists: episode complete, shutting down"
  quit(0)

proc abortEpisodeWithFailure(runtimeConfig: RuntimeConfig, failurePayload: JsonNode) =
  ## Unscored explicit termination on exhausted model or transport failure.
  ## Never settles the sim, never sends final.scores, and never writes
  ## resultsUri or replayUri; emits only raw failure evidence and exits 1.
  withLock stateLock:
    if state.finished:
      return
    state.finished = true
    state.trajectory.failNativeTrajectory(failurePayload)
    let errorFrame = $ %*{
      "type": "error",
      "done": true,
      "reason": failurePayload["reason"],
      "failure_kind": failurePayload["failure_kind"],
      "failures": failurePayload["failures"]
    }
    for _, socket in state.playerSockets:
      socket.send(errorFrame)
    for socket in state.globalSockets:
      socket.send(errorFrame)

  sleep(500)
  let failureUri = getEnv("COGAME_PLAYER_FAILURE_URI").strip()
  if failureUri.len > 0:
    echo "cogchemists: writing player failure artifact"
    writeArtifact(
      failureUri, $failurePayload, "application/json",
      "COGAME_PLAYER_FAILURE_METHOD"
    )
  echo "cogchemists: episode aborted unscored (", failurePayload["failure_kind"].getStr(), ")"
  quit(1)

const PlayBudgetFraction* = 0.6
  ## Share of the platform's episode timeout spent playing. The rest covers
  ## container start, player connects, and writing the artifacts — the part
  ## that must never be the thing that runs out of time.

proc runGame(runtimeConfig: RuntimeConfig) {.gcsafe.} =
  {.gcsafe.}:
    let config = state.config
    let gameStart = epochTime()
    let connectDeadline = gameStart + config.playerConnectTimeoutSeconds

    while epochTime() < connectDeadline:
      var allConnected = false
      withLock stateLock:
        allConnected = state.playerSockets.len >= config.tokens.len
        for registered in state.registered:
          allConnected = allConnected and registered
      if allConnected:
        break
      sleep(200)

    withLock stateLock:
      state.started = true
      echo "cogchemists: starting with ", state.playerSockets.len, "/",
        config.tokens.len, " players connected"
      state.broadcastLocked()

    let client = newLlmClient(config)

    ## The platform kills the episode at its timeout and keeps nothing, so
    ## play inside a fraction of it. The hosted dispatcher hands the timeout
    ## only to its own worker sidecar, NOT to the game container, so when
    ## the env is silent assume the configured platform default rather than
    ## playing open-ended.
    let hostedTimeout = getEnv("COWORLD_TIMEOUT_SECONDS", "").strip()
    var timeoutSeconds =
      if hostedTimeout.len > 0:
        try: parseFloat(hostedTimeout) except ValueError: 0.0
      else: 0.0
    if timeoutSeconds <= 0.0:
      timeoutSeconds = config.episodeTimeoutSeconds.float
    let playDeadline =
      if timeoutSeconds > 0.0: gameStart + timeoutSeconds * PlayBudgetFraction
      else: 0.0
    if playDeadline > 0.0:
      echo "cogchemists: episode timeout ", timeoutSeconds.int, "s (",
        (if hostedTimeout.len > 0: "from env" else: "assumed"),
        "); playing until ", (timeoutSeconds * PlayBudgetFraction).int, "s"

    while true:
      var simCopy: Sim
      var seats: seq[int]
      var prompts: seq[string]
      var scripted: seq[ScriptKind]
      var order: seq[int]
      var structural = false
      var privateViews: seq[JsonNode]
      withLock stateLock:
        if state.sim.done:
          break
        if state.sim.needsAdvance():
          ## Round opens, phase opens, the exhibition and the settle each
          ## emit exactly one event, so every frame is distinct.
          state.sim.advance()
          state.broadcastLocked()
          structural = true
        else:
          seats = state.sim.pendingSeats()
          simCopy = state.sim
          prompts = state.prompts
          scripted = state.scripted
          order = initiativeOrder(state.sim.round)
          if not state.trajectory.isNil:
            for seat in seats:
              privateViews.add(%*{"view": state.sim.observationJson(seat),
                "operator_prompt": prompts[seat], "round": state.sim.round,
                "phase": $state.sim.phase})
          echo "cogchemists: round ", state.sim.round + 1, "/", config.rounds,
            " ", $state.sim.phase, " at ", (epochTime() - gameStart).int, "s"
      if structural:
        continue

      ## The deadline is tested BEFORE every batch, i.e. only ever between
      ## phases. Past it the exhibition still runs, so no seat keeps
      ## unearned publication credit.
      if playDeadline > 0.0 and epochTime() > playDeadline:
        withLock stateLock:
          echo "cogchemists: episode deadline reached after ",
            state.sim.roundsPlayed, "/", config.rounds,
            " rounds; holding the exhibition and ending early"
          state.sim.endEarly()
          state.broadcastLocked()
        break

      if seats.len == 0:
        sleep(50)
        continue

      ## External players decide over the same phase snapshot as prompt and
      ## scripted seats. Send all observations before waiting for any reply.
      var nonExternalSeats: seq[int]
      var isExternal = newSeq[bool](seats.len)
      withLock stateLock:
        inc state.decisionId
        for index, seat in seats:
          if state.external[seat]:
            isExternal[index] = true
            state.pending[seat] = true
            state.accepted[seat] = false
            state.lastRawInput[seat] = ""
            state.lastRejection[seat] = ""
            if state.playerSockets.hasKey(seat):
              state.playerSockets[seat].send($ %*{
                "type": "observation", "id": state.decisionId,
                "observation": simCopy.observationJson(seat)
              })
          else:
            nonExternalSeats.add(seat)

      ## Prompt and explicitly scripted seats run in one parallel batch.
      ## External seats never receive a scripted fallback.
      let nonExternalOutcomes =
        if nonExternalSeats.len > 0:
          client.decideAll(simCopy, nonExternalSeats, prompts, scripted, state.selection)
        else:
          @[]

      let deadline = epochTime() + config.llmTimeoutSeconds.float
      while epochTime() < deadline:
        var waiting = false
        withLock stateLock:
          for seat in seats:
            if state.pending[seat] and not state.accepted[seat] and
                state.playerSockets.hasKey(seat):
              waiting = true
        if not waiting:
          break
        sleep(20)

      var outcomes = newSeq[DecisionOutcome](seats.len)
      var nonExtPos = 0
      withLock stateLock:
        for index, seat in seats:
          if isExternal[index]:
            if state.pending[seat] and state.accepted[seat]:
              outcomes[index] = DecisionOutcome(
                ok: true,
                action: state.actions[seat],
                fromScript: false,
                rawReply: state.lastRawInput[seat]
              )
            else:
              let hadRejection = state.lastRejection[seat].len > 0
              let err =
                if hadRejection: state.lastRejection[seat]
                else: "external player failed to submit a valid legal action"
              outcomes[index] = DecisionOutcome(
                ok: false,
                fromScript: false,
                failureKind: (if hadRejection: fkMalformedPlayerData else: fkExternalTimeout),
                error: err,
                rawReply: state.lastRawInput[seat]
              )
            state.pending[seat] = false
          else:
            outcomes[index] = nonExternalOutcomes[nonExtPos]
            inc nonExtPos

      var failureList = newJArray()
      var primaryKind = fkNone
      for index, seat in seats:
        if not outcomes[index].ok:
          let kind =
            if outcomes[index].failureKind != fkNone: outcomes[index].failureKind
            else: fkMalformedPlayerData
          if primaryKind == fkNone or kind == fkTransportFailure:
            primaryKind = kind
          failureList.add(%*{
            "slot": seat,
            "name": config.players[seat].name,
            "alias": simCopy.names[seat],
            "failure_kind": $kind,
            "error": outcomes[index].error,
            "raw_response": outcomes[index].rawReply
          })

      if failureList.len > 0:
        let failurePayload = %*{
          "reason": $primaryKind,
          "failure_kind": $primaryKind,
          "round": simCopy.round,
          "phase": $simCopy.phase,
          "failures": failureList
        }
        withLock stateLock:
          for index, seat in seats:
            if not outcomes[index].ok and not state.trajectory.isNil:
              let attempts =
                if seat < client.nativeAttempts.len: client.nativeAttempts[seat]
                else: @[]
              state.trajectory.recordFailedDecision(config, state.selection,
                simCopy.round * 2 + ord(simCopy.phase), seat, privateViews[index],
                outcomes[index].error, outcomes[index].rawReply, attempts)
        abortEpisodeWithFailure(runtimeConfig, failurePayload)
        return

      withLock stateLock:
        for seat in order:
          var position = -1
          for index, candidate in seats:
            if candidate == seat:
              position = index
          if position < 0:
            continue
          let outcome = outcomes[position]
          let decision = outcome.action
          let wasScripted = outcome.fromScript
          var engineRejected = false
          var resolutionReason = ""
          var applied = decision
          if not config.talk: applied.say = ""
          try:
            state.sim.applyAct(seat, decision, wasScripted)
            echo "cogchemists: ", state.sim.names[seat], " plays ",
              showAct(decision)
          except CogchemistsError as error:
            ## An earlier seat in this same phase undid the move. Recorded
            ## as rejected and degraded to pass per simultaneous-phase rules.
            echo "cogchemists: ", state.sim.names[seat], " rejected (",
              error.msg, "); passing"
            state.sim.applyRejection(seat, decision, error.msg, wasScripted)
            engineRejected = true
            resolutionReason = error.msg
            applied = newAction("pass", say = applied.say, notes = applied.notes)
          if not state.trajectory.isNil:
            let attempts =
              if seat < client.nativeAttempts.len: client.nativeAttempts[seat]
              else: @[]
            state.trajectory.recordAppliedDecision(config, state.selection,
              simCopy.round * 2 + ord(simCopy.phase), seat, privateViews[position],
              applied.actionReply(), wasScripted, engineRejected, resolutionReason,
              attempts)
        state.broadcastLocked()

      if config.turnDelayMs > 0:
        sleep(config.turnDelayMs)

    ## Drain any structural events the loop still owes (the exhibition and
    ## the end frame) before the final broadcast.
    while true:
      var more = false
      withLock stateLock:
        if state.sim.needsAdvance():
          state.sim.advance()
          state.broadcastLocked()
          more = true
      if not more:
        break

    if config.turnDelayMs > 0:
      sleep(config.turnDelayMs)
    finishEpisode(runtimeConfig)

var gameThread: Thread[RuntimeConfig]

proc serveFile(request: Request, path, contentType: string) =
  if fileExists(path):
    var headers: HttpHeaders
    headers["Content-Type"] = contentType
    request.respond(200, headers, readFile(path))
  else:
    request.respond(404)

proc htmlHandler(name: string): RequestHandler =
  proc handler(request: Request) {.gcsafe.} =
    {.gcsafe.}:
      serveFile(request, clientDir() / name, "text/html; charset=utf-8")
  handler

proc assetHandler(request: Request) {.gcsafe.} =
  {.gcsafe.}:
    let name = request.pathParams["name"]
    if "/" in name or "\\" in name or name.startsWith("."):
      request.respond(404)
      return
    let contentType =
      if name.endsWith(".png"): "image/png"
      elif name.endsWith(".ttf"): "font/ttf"
      else: "application/octet-stream"
    serveFile(request, dataDir() / name, contentType)

proc rendererHandler(request: Request) {.gcsafe.} =
  {.gcsafe.}:
    serveFile(
      request, clientDir() / "renderer.js",
      "application/javascript; charset=utf-8"
    )

proc chromeCssHandler(request: Request) {.gcsafe.} =
  {.gcsafe.}:
    serveFile(
      request, clientDir() / "chrome.css",
      "text/css; charset=utf-8"
    )

proc healthzHandler(request: Request) {.gcsafe.} =
  var headers: HttpHeaders
  headers["Content-Type"] = "application/json"
  request.respond(200, headers, """{"ok": true}""")

proc playerUpgradeHandler(request: Request) {.gcsafe.} =
  {.gcsafe.}:
    let slotText = request.queryParams["slot"]
    let token = request.queryParams["token"]
    var slot = -1
    try:
      slot = parseInt(slotText)
    except ValueError:
      discard
    var authorized = false
    withLock stateLock:
      authorized = slot >= 0 and slot < state.config.tokens.len and
        state.config.tokens[slot] == token
    if not authorized:
      request.respond(401)
      return
    let websocket = request.upgradeToWebSocket()
    withLock stateLock:
      state.playerSockets[slot] = websocket
      state.socketSlots[websocket] = slot
      echo "cogchemists: player slot ", slot, " connected (",
        state.playerSockets.len, "/", state.config.tokens.len, ")"
      websocket.send($ %*{
        "type": "welcome",
        "protocol": "cogchemists.player.v2",
        "slot": slot,
        "name": state.sim.names[slot],
        "rounds": state.config.rounds
      })

proc globalUpgradeHandler(request: Request) {.gcsafe.} =
  {.gcsafe.}:
    let websocket = request.upgradeToWebSocket()
    withLock stateLock:
      state.globalSockets.incl(websocket)
      websocket.send($state.snapshotJson())

proc replayUpgradeHandler(request: Request) {.gcsafe.} =
  {.gcsafe.}:
    let websocket = request.upgradeToWebSocket()
    if replayPayloadGlobal.len > 0:
      websocket.send(replayPayloadGlobal)

proc websocketHandler(
  websocket: WebSocket,
  event: WebSocketEvent,
  message: Message
) {.gcsafe.} =
  {.gcsafe.}:
    case event
    of OpenEvent:
      discard
    of MessageEvent:
      ## mummy hands Ping frames to the application instead of answering
      ## them itself; the platform's certifier pings /global to check the
      ## game is alive, so an unanswered ping fails certification.
      if message.kind == Ping:
        websocket.send(message.data, Pong)
        return
      if message.kind != TextMessage:
        return
      var slot = -1
      withLock stateLock:
        slot = state.socketSlots.getOrDefault(websocket, -1)
      if slot < 0:
        return
      try:
        let payload = parseJson(message.data)
        if payload{"type"}.getStr() == "prompt":
          var prompt = payload{"prompt"}.getStr()
          ## Rune boundary, not byte: the prompt is quoted into logs.
          if prompt.runeLen > MaxPromptLen:
            prompt = prompt.runeSubStr(0, MaxPromptLen)
          let node = payload{"scripted"}
          let scripted =
            if node.isNil or node.kind == JNull: skNone
            elif node.kind == JString: parseScriptKind(node.getStr())
            else:
              raise newException(CogchemistsError,
                "scripted must be a string (\"assayer\", \"quack\", or \"\")")
          withLock stateLock:
            state.prompts[slot] = prompt
            state.scripted[slot] = scripted
            state.external[slot] = false
            state.registered[slot] = true
          echo "cogchemists: slot ", slot, " delivered a prompt (",
            prompt.len, " chars",
            (if scripted != skNone: ", scripted " & $scripted else: ""), ")"
        elif payload{"type"}.getStr() == "register" and
            payload["control"].getStr() == "external":
          withLock stateLock:
            state.external[slot] = true
            state.registered[slot] = true
          echo "cogchemists: slot ", slot, " registered external control"
        elif payload{"type"}.getStr() == "action":
          let id = payload["id"].getInt()
          let move = payload["move"].getStr()
          var rejectReason = ""
          withLock stateLock:
            if state.external[slot] and state.pending[slot] and
                state.decisionId == id and not state.accepted[slot]:
              state.lastRawInput[slot] = message.data
              for action in state.sim.legalActs(slot):
                if showAct(action) == move:
                  state.actions[slot] = action
                  state.actions[slot].say = payload{"say"}.getStr()
                  state.actions[slot].notes = payload{"notes"}.getStr()
                  state.accepted[slot] = true
                  break
              if not state.accepted[slot]:
                rejectReason = "illegal move: " & move
                state.lastRejection[slot] = rejectReason
          if rejectReason.len > 0:
            websocket.send($ %*{
              "type": "rejected",
              "id": id,
              "reason": rejectReason,
              "move": move
            })
      except CatchableError as error:
        echo "cogchemists: ignoring bad player frame: ", error.msg
    of ErrorEvent:
      discard
    of CloseEvent:
      withLock stateLock:
        if websocket in state.socketSlots:
          let slot = state.socketSlots[websocket]
          state.socketSlots.del(websocket)
          if state.playerSockets.getOrDefault(slot) == websocket:
            state.playerSockets.del(slot)
        state.globalSockets.excl(websocket)

proc buildRouter(replayMode: bool): Router =
  ## Both /client/player and /client/global serve real pages and are
  ## registered BEFORE any catch-all asset route: the certifier probes them
  ## before the player pods start, and neither may open the player socket.
  result.get("/healthz", healthzHandler)
  result.get("/client/global", htmlHandler("global.html"))
  result.get("/client/player", htmlHandler("player.html"))
  result.get("/client/replay", htmlHandler("replay.html"))
  result.get("/client/renderer.js", rendererHandler)
  result.get("/client/chrome.css", chromeCssHandler)
  result.get("/client/assets/@name", assetHandler)
  result.get("/global", globalUpgradeHandler)
  result.get("/replay", replayUpgradeHandler)
  if not replayMode:
    result.get("/player", playerUpgradeHandler)

proc configFromReplay*(payload: JsonNode): GameConfig =
  result = defaultGameConfig()
  result.rounds = payload["config"]{"rounds"}.getInt(6)
  result.seed = payload["config"]{"seed"}.getInt(0)
  result.talk = payload["config"]{"talk"}.getBool(true)
  result.artifacts = payload["config"]{"artifacts"}.getBool(true)
  ## The replay carries the episode's fitted cap; never re-fit it. The
  ## chemistry and every draw are re-derived from the seed.
  result.sampled = true
  for name in payload["names"]:
    result.players.add(PlayerConfig(name: name.getStr()))

proc runReplayServer*(runtimeConfig: RuntimeConfig) =
  ## Replay mode: parse the recorded replay, precompute the scrub states,
  ## and serve the viewer until the platform tears the container down.
  let payload = parseJson(runtimeConfig.replay)
  let config = configFromReplay(payload)
  var events: seq[GameEvent]
  for node in payload["events"]:
    events.add(eventFromJson(node))
  var enriched = %*{
    "type": "replay",
    "protocol": payload{"protocol"}.getStr("cogchemists.replay.v1"),
    "names": payload["names"],
    "policyNames": payload{"policyNames"},
    "config": payload["config"],
    "events": payload["events"],
    "results": payload{"results"},
    "states": statesFromEvents(config, events)
  }
  replayPayloadGlobal = $enriched

  let router = buildRouter(replayMode = true)
  gameServer = newServer(router, websocketHandler)
  echo "cogchemists: replay mode on ", runtimeConfig.host, ":",
    runtimeConfig.port
  gameServer.serve(Port(runtimeConfig.port), runtimeConfig.host)

proc runGameServer*(config: GameConfig, runtimeConfig: RuntimeConfig) =
  if config.tokens.len != config.players.len:
    raise newException(CogchemistsError, "tokens and players must align")
  state.config = config
  state.selection = promptSelection(if runtimeConfig.config.len == 0:
    newJObject() else: parseJson(runtimeConfig.config))
  state.sim = initSim(config)
  state.trajectory = initNativeTrajectory(config)
  state.prompts = newSeq[string](config.players.len)
  state.scripted = newSeq[ScriptKind](config.players.len)
  state.external = newSeq[bool](config.players.len)
  state.registered = newSeq[bool](config.players.len)
  state.pending = newSeq[bool](config.players.len)
  state.accepted = newSeq[bool](config.players.len)
  state.actions = newSeq[Action](config.players.len)
  state.lastRawInput = newSeq[string](config.players.len)
  state.lastRejection = newSeq[string](config.players.len)
  runtimeConfigGlobal = runtimeConfig

  let router = buildRouter(replayMode = false)
  gameServer = newServer(router, websocketHandler)
  createThread(gameThread, runGame, runtimeConfig)
  echo "cogchemists: serving on ", runtimeConfig.host, ":",
    runtimeConfig.port
  gameServer.serve(Port(runtimeConfig.port), runtimeConfig.host)
