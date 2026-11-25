## The scripted baselines must play whole episodes without ever proposing an
## illegal action — they are both the no-credentials fallback (offline
## certification, the docker smoke) and fieldable policies. The assayer must
## also actually be worth beating, and the reply parser must read every shape
## the schema documents.

import std/[json, monotimes, os, strutils, times, unicode, unittest]
import cogchemists/[llm, sim, prompts]

const
  EpisodeBudgetMs = when defined(release): 2000 else: 20000
  DisabledBudgetMs = when defined(release): 5000 else: 30000

proc fixture(seed: int, rounds = 6, talk = true): GameConfig =
  result = defaultGameConfig()
  result.seed = seed
  result.rounds = rounds
  result.talk = talk
  result.minBatchSpacingMs = 0
  result.sampled = true
  for index in 0 ..< Seats:
    result.players.add(PlayerConfig(name: "P" & $(index + 1)))
    result.tokens.add("t" & $index)

proc drive(sim: var Sim) =
  while (not sim.done) and sim.needsAdvance():
    sim.advance()

type Tally = object
  unsolvedPublishes: int
  falseAtExhibition: int
  burned: int

proc playScripted(config: GameConfig, kinds: array[Seats, ScriptKind],
    tally: var Tally): Sim =
  result = initSim(config)
  result.drive()
  while not result.done:
    for seat in initiativeOrder(result.round):
      let decision = scriptedAction(result, seat, kinds[seat])
      ## The bounded/legal assertion: the bot's action must be a member of
      ## the legal set AT THE MOMENT IT IS PLAYED, not merely survive
      ## applyAct. doAssert, not check: this runs outside the test body, and
      ## a failure has to abort with the offending move rather than tick a
      ## counter nobody reads.
      doAssert showAct(decision) in result.legalMoves(seat),
        "illegal scripted move: " & showAct(decision) & " (seat " & $seat &
        ", round " & $result.round & ", " & $result.phase & ")"
      doAssert decision.say.len == 0, "a scripted seat never talks"
      doAssert decision.notes.len == 0, "a scripted seat never takes notes"
      if decision.action == "publish" and
          not result.grids[seat].solved(decision.a):
        inc tally.unsolvedPublishes
      ## applyAct raises on anything illegal and would fail this test.
      result.applyAct(seat, decision, true)
      doAssert result.seats[seat].coin >= 0, "coin went negative"
      doAssert result.seats[seat].hand.len <= HandCap, "hand cap exceeded"
    result.drive()
  for seat in 0 ..< Seats:
    tally.falseAtExhibition += result.falseTheories[seat]
    tally.burned += result.burnedCount[seat]

proc meanScore(sim: Sim): float =
  for seat in 0 ..< Seats:
    result += sim.score(seat)
  result / Seats.float

suite "legality and boundedness":
  test "every scripted table completes without an illegal action":
    let mixes: array[3, array[Seats, ScriptKind]] = [
      [skAssayer, skAssayer, skAssayer, skAssayer],
      [skQuack, skQuack, skQuack, skQuack],
      [skAssayer, skQuack, skAssayer, skQuack]
    ]
    for seed in [1, 7, 11, 42]:
      for kinds in mixes:
        var tally = Tally()
        let started = getMonoTime()
        let sim = playScripted(fixture(seed), kinds, tally)
        let elapsed = (getMonoTime() - started).inMilliseconds
        check sim.done
        check sim.reason == "complete"
        check sim.roundsPlayed == sim.config.rounds
        ## No seat may name a card it does not hold, and no ingredient may
        ## carry two standing seals.
        var standing: array[Ingredients, int]
        for seal in sim.seals:
          if seal.status == sealStanding:
            inc standing[seal.ingredient]
        for ingredient in 0 ..< Ingredients:
          check standing[ingredient] <= 1
        check elapsed < EpisodeBudgetMs

suite "baseline behaviour":
  test "the assayer publishes only what it has solved":
    for seed in [1, 7, 11, 42]:
      var tally = Tally()
      let sim = playScripted(fixture(seed),
        [skAssayer, skAssayer, skAssayer, skAssayer], tally)
      ## A solved ingredient's one candidate must BE the truth, so a
      ## disciplined publisher can never be proved wrong.
      check tally.unsolvedPublishes == 0
      for seat in 0 ..< Seats:
        check sim.falseTheories[seat] == 0
        check sim.burnedCount[seat] == 0

  test "the quack publishes on hunches and pays for it":
    var unsolved = 0
    var punished = 0
    for seed in [1, 7, 11, 42]:
      var tally = Tally()
      discard playScripted(fixture(seed),
        [skQuack, skQuack, skQuack, skQuack], tally)
      check tally.unsolvedPublishes >= 1
      unsolved += tally.unsolvedPublishes
      punished += tally.falseAtExhibition + tally.burned
    echo "  quack: ", unsolved, " unsolved seals, ", punished,
      " proved false or burned"
    check punished >= 1

  test "an assayer table beats a quack table on the same seed":
    for seed in [1, 7, 11, 42]:
      var tallyA = Tally()
      var tallyQ = Tally()
      let assayers = playScripted(fixture(seed),
        [skAssayer, skAssayer, skAssayer, skAssayer], tallyA)
      let quacks = playScripted(fixture(seed),
        [skQuack, skQuack, skQuack, skQuack], tallyQ)
      echo "  seed ", seed, ": assayer mean ", meanScore(assayers),
        " vs quack mean ", meanScore(quacks)
      check meanScore(assayers) > meanScore(quacks)

suite "explicit baselines and explicit failure (no implicit fallback)":
  test "explicitly scripted seats complete with no network wait":
    for envVar in ["COWORLD_LLM_ENDPOINT", "ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY_URI",
        "AWS_ENDPOINT_URL_BEDROCK_RUNTIME", "AWS_BEARER_TOKEN_BEDROCK"]:
      delEnv(envVar)
    var config = fixture(3)
    config.minBatchSpacingMs = 10_000
    let client = newLlmClient(config)
    check client.disabled
    check client.minBatchSpacingMs == 0
    var sim = initSim(config)
    sim.drive()
    let started = getMonoTime()
    let scripted = @[skAssayer, skQuack, skAssayer, skQuack]
    while not sim.done:
      let seats = sim.pendingSeats()
      check seats.len == Seats
      let outcomes = client.decideAll(sim, seats,
        newSeq[string](Seats), scripted, PromptSelection())
      check outcomes.len == Seats
      for outcome in outcomes:
        check outcome.ok
        check outcome.fromScript
      for seat in initiativeOrder(sim.round):
        var position = -1
        for index, candidate in seats:
          if candidate == seat:
            position = index
        try:
          sim.applyAct(seat, outcomes[position].action, outcomes[position].fromScript)
        except CogchemistsError as error:
          sim.applyRejection(seat, outcomes[position].action, error.msg,
            outcomes[position].fromScript)
      sim.drive()
    let elapsed = (getMonoTime() - started).inMilliseconds
    echo "  explicitly scripted 6-round episode: ", elapsed, " ms"
    check sim.reason == "complete"
    check elapsed < DisabledBudgetMs
    for event in sim.events:
      if event.kind == evAct:
        check event.scripted

  test "a prompt seat without LLM credentials fails explicitly without fallback":
    for envVar in ["COWORLD_LLM_ENDPOINT", "ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY_URI",
        "AWS_ENDPOINT_URL_BEDROCK_RUNTIME", "AWS_BEARER_TOKEN_BEDROCK"]:
      delEnv(envVar)
    var config = fixture(3)
    let client = newLlmClient(config)
    check client.disabled
    var sim = initSim(config)
    sim.drive()
    let seats = sim.pendingSeats()
    let outcomes = client.decideAll(sim, seats,
      newSeq[string](Seats), @[skNone, skAssayer, skNone, skQuack], PromptSelection())
    check outcomes.len == Seats
    check not outcomes[0].ok
    check not outcomes[0].fromScript
    check outcomes[0].failureKind == fkMissingCredentials
    check "no LLM credentials" in outcomes[0].error
    check outcomes[1].ok and outcomes[1].fromScript
    check not outcomes[2].ok
    check not outcomes[2].fromScript
    check outcomes[2].failureKind == fkMissingCredentials
    check outcomes[3].ok and outcomes[3].fromScript

  test "a seat whose reply fails twice fails explicitly without scripted fallback":
    putEnv("AWS_ENDPOINT_URL_BEDROCK_RUNTIME", "http://127.0.0.1:1")
    putEnv("AWS_BEARER_TOKEN_BEDROCK", "not-a-real-token")
    var config = fixture(4)
    config.llmTimeoutSeconds = 2
    let client = newLlmClient(config)
    delEnv("AWS_ENDPOINT_URL_BEDROCK_RUNTIME")
    delEnv("AWS_BEARER_TOKEN_BEDROCK")
    check not client.disabled
    var sim = initSim(config)
    sim.drive()
    let seats = sim.pendingSeats()
    let outcomes = client.decideAll(sim, seats,
      newSeq[string](Seats), newSeq[ScriptKind](Seats), PromptSelection())
    check outcomes.len == Seats
    for outcome in outcomes:
      check not outcome.ok
      check not outcome.fromScript
      check outcome.failureKind == fkTransportFailure
      check outcome.error.len > 0

  test "parseScriptKind accepts only canonical values and rejects aliases":
    check parseScriptKind("") == skNone
    check parseScriptKind("assayer") == skAssayer
    check parseScriptKind("quack") == skQuack
    for alias in ["1", "true", "yes", "scientist", "careerist", "ASSAYER", "bogus"]:
      expect CogchemistsError:
        discard parseScriptKind(alias)

suite "reply parsing":
  test "the documented shapes parse and the caps hold":
    var sim = initSim(fixture(5))
    sim.drive()
    sim.seats[0].hand = @[0, 6]
    let reply = parseJson("""{"action":"test_student","a":"Nightcap",
      "b":"Sunmoss","signature":"","artifact":"","say":"a hunch",
      "notes":"keep going"}""")
    let action = parseReply(sim, 0, reply)
    check action.action == "test_student"
    check action.a == 0
    check action.b == 6
    check action.say == "a hunch"
    check sim.checkAct(0, action) == ""
    ## Unused fields may be omitted.
    let minimal = parseReply(sim, 0, parseJson("""{"action":"forage"}"""))
    check minimal.action == "forage"
    check sim.checkAct(0, minimal) == ""

  test "exact canonical ingredient names resolve and aliases are rejected":
    for index, name in IngredientNames:
      check resolveIngredient(name) == index
    check resolveIngredient("  Nightcap  ") == 0
    for alias in ["nightcap", "NIGHT", "0", "7", "Rime", "widows salt", "wid", "xyzzy", "8", ""]:
      check resolveIngredient(alias) == -1

  test "every free-text field is capped on rune boundaries":
    var sim = initSim(fixture(6))
    sim.drive()
    var payload = %*{
      "action": "pass",
      "a": "", "b": "", "signature": "", "artifact": "",
      "say": "é".repeat(400),
      "notes": "ü".repeat(900)
    }
    let action = parseReply(sim, 0, payload)
    check action.say.runeLen == MaxSayLen
    check action.notes.runeLen == MaxNotesLen
    check action.say.validateUtf8() == -1
    check action.notes.validateUtf8() == -1
    check cleanText("a".repeat(100), MaxActionLen).runeLen == MaxActionLen

  test "wrong-phase, aliases, unresolvable, malformed and unaffordable are invalid":
    var sim = initSim(fixture(7))
    sim.drive()
    check sim.phase == phLab
    ## Prompt example matches LAB phase, not MARKET.
    let labPrompt = sim.userPrompt(0, "")
    check "\"action\":\"test_student\"" in labPrompt
    check "\"action\":\"publish\"" notin labPrompt
    ## A market action in the lab phase.
    expect CogchemistsError:
      discard parseReply(sim, 0, %*{"action": "publish", "a": "Nightcap",
        "signature": "R+G-B+"})
    ## Hyphenated, spaced, and uppercased verb aliases are rejected.
    for badVerb in ["test-student", "test student", "TEST_STUDENT", "Pass"]:
      expect CogchemistsError:
        discard parseReply(sim, 0, %*{"action": badVerb, "a": "Nightcap", "b": "Sunmoss"})
    ## An ingredient that does not resolve.
    expect CogchemistsError:
      discard parseReply(sim, 0, %*{"action": "transmute", "a": "Moonjuice"})
    expect CogchemistsError:
      discard parseReply(sim, 0, %*{"action": "transmute", "a": "0"})
    ## No action at all.
    expect CogchemistsError:
      discard parseReply(sim, 0, %*{"say": "hello"})
    ## A malformed or non-canonical signature, in the phase where publish is legal.
    for seat in initiativeOrder(sim.round):
      sim.applyAct(seat, newAction("pass"), true)
    sim.drive()
    check sim.phase == phMarket
    let marketPrompt = sim.userPrompt(0, "")
    check "\"action\":\"publish\"" in marketPrompt
    for badSig in ["R+G-", "+-+", "r+g-b+", "B+G-R+"]:
      expect CogchemistsError:
        discard parseReply(sim, 0, %*{"action": "publish", "a": "Nightcap",
          "signature": badSig})
    for badArt in ["telescope", "MORTAR", "Press"]:
      expect CogchemistsError:
        discard parseReply(sim, 0, %*{"action": "buy", "artifact": badArt})
    ## Unaffordable: the schema reads it, and the RULES refuse it — which is
    ## what decideAll probes with before accepting a reply.
    sim.seats[0].coin = 0
    let broke = parseReply(sim, 0, %*{"action": "publish", "a": "Nightcap",
      "signature": "R+G-B+"})
    check sim.checkAct(0, broke) == "no_coin"

  test "raw JSON and outer markdown JSON fences parse; surrounding prose is rejected":
    let raw = """{"action":"pass","notes":"ok"}"""
    check extractJsonObject(raw)["action"].getStr() == "pass"
    let fencedJson = "```json\n{\"action\":\"pass\",\"notes\":\"ok\"}\n```"
    check extractJsonObject(fencedJson)["action"].getStr() == "pass"
    let fencedPlain = "```\n{\"action\":\"forage\"}\n```"
    check extractJsonObject(fencedPlain)["action"].getStr() == "forage"
    for bad in [
      "```json\n{\"action\":\"pass\"}\n```\nI hope that helps!",
      "Here is my move:\n{\"action\":\"pass\"}",
      "{\"action\":\"pass\"} trailing text",
      "I am afraid I cannot do that.",
      "[1, 2, 3]"
    ]:
      expect CogchemistsError:
        discard extractJsonObject(bad)

  test "error text quoted into the log is cut on a rune boundary":
    ## The reply itself never reaches the replay, but it does reach stdout,
    ## and a byte cut through a multi-byte character leaves invalid UTF-8
    ## there (design note, Reply schema: the rune rule applies "to any error
    ## text quoted into the log").
    var caught = ""
    try:
      discard extractJsonObject("é".repeat(400))
    except CogchemistsError as error:
      caught = error.msg
    check caught.len > 0
    check caught.validateUtf8() == -1
    check caught.runeLen < 400
