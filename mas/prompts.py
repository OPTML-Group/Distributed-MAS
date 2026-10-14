"""Role prompts for the two MAS architectures of the paper (Fig. 2a, App. C.1).

    sequential_v2           S-MAS  Solver -> Verifier -> Corrector -> Decider
    hierarchical            H-MAS  Analyzer 1/2/3 (independent) -> Synthesizer
    sequential_v2s          S-MAS with a safety-aware Verifier   } hardened collection
    hierarchical_v2s        H-MAS with a safety-scoped Analyzer  } prompts (teacher)

Internally every architecture uses the same four role keys

    planner  critic  refiner  judger

so that a role maps to the same position in both chains:

    S-MAS:  planner=Solver     critic=Verifier    refiner=Corrector   judger=Decider
    H-MAS:  planner=Analyzer1  critic=Analyzer2   refiner=Analyzer3   judger=Synthesizer

The prompt text is reproduced verbatim from the runs reported in the paper.
Every agent uses the same system message; the final agent (`judger`) is the one
whose message is the system output.
"""

# Answer-format instruction, chosen by task. Shared by every builder below and
# by the single-agent prompt so that all arms are asked for the same format.
_NUMERIC_TASKS = ("gsm8k", "math500")
_OPTION_TASKS = ("bbh", "mmlu_pro", "musr")
_ABCD_TASKS = ("gpqa",)


def answer_format_instruction(task) -> str:
    """The sentence that tells the model how to emit a parseable final answer."""
    if task in _ABCD_TASKS:
        return ("Your final answer must be selected from A,B,C,D. For example "
                "\\boxed{A}. Do not add any other contents inside the box.\n"
                "Now, reason step by step and output the final answer inside "
                "\\boxed{YOUR_FINAL_ANSWER}.")
    if task in _OPTION_TASKS:
        # Option counts vary (MMLU-Pro has ten; several BBH subtasks are yes/no
        # or true/false with no listed options), so name the form, not A-D.
        return ("Your final answer must be the single option letter of the correct "
                "choice, for example \\boxed{B}. If the question is a yes/no or "
                "true/false question with no listed options, answer with that word, "
                "for example \\boxed{Yes} or \\boxed{False}. Do not add any other "
                "contents inside the box.\n"
                "Now, reason step by step and output the final answer inside "
                "\\boxed{YOUR_FINAL_ANSWER}.")
    return ("Now, reason step by step and output the final answer inside "
            "\\boxed{YOUR_FINAL_ANSWER}.")


def build_agent_messages_sequential_v2_text_mas(role: str, question: str, context: str = "",
                                                method=None, args=None):
    """S-MAS (Fig. 2a, App. C.1): Solver -> Verifier -> Corrector -> Decider.

    Internal role keys are planner / critic / refiner / judger. The Solver
    writes a complete numbered solution; the Verifier only reports errors (or
    `NO ERROR FOUND`); the Corrector fixes the flagged steps; the Decider
    issues the final answer without re-solving.
    """
    system_message = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    assert method in ["text_mas"], "only for text_mas method"
    ctx = context[: args.text_mas_context_length]
    fmt = answer_format_instruction(getattr(args, "task", None))

    if role == "planner":
        user_content = f"""You are the Solver, the first agent in a multi-agent system. Solve the question completely and carefully.

## Question
{question}

Work through the problem step by step and commit to a specific final answer. Number your steps, and state the value or fact each step establishes, so a later agent can check any single step in isolation. Do not hedge or offer alternatives — later agents will correct you if you are wrong.

{fmt}"""

    elif role == "critic":
        user_content = f"""You are the Verifier. Another agent has produced a candidate solution. Your only job is to find errors in it.

## Question
{question}

## Candidate solution
{ctx}

Check, in this order: (1) does it answer the question that was actually asked; (2) is the method/formula the right one; (3) is each arithmetic or algebraic step correct; (4) are units, signs, and rounding right; (5) is the final answer in the form the question demands.

Rules:
- Do NOT restate or summarize the candidate solution. Assume the next agent can see it.
- Report only substantive errors that change the answer. Ignore style.
- For each error, name the step number and say what is wrong and what it should be.
- If you check it and find nothing wrong, reply with exactly: NO ERROR FOUND

## Your output
Either `NO ERROR FOUND`, or a short list of errors in the form:
Step <n>: <what is wrong> -> <what it should be>"""

    elif role == "refiner":
        user_content = f"""You are the Corrector. You are given a question, a candidate solution, and a verifier's error report.

## Question
{question}

## Candidate solution and verifier report
{ctx}

If the verifier reported NO ERROR FOUND, restate the candidate's final answer unchanged and stop — do not re-derive it, and do not second-guess a solution that survived verification.

Otherwise, fix only the steps the verifier flagged, carry the corrections through to the end, and give the corrected final answer. Do not rewrite steps that were not flagged.

{fmt}"""

    else:   # judger / sink
        user_content = f"""You are the Decider, the final agent. The solving and checking have already been done by the agents before you.

## Question
{question}

## Prior agents' work
{ctx}

Do not solve the problem again from scratch. Your job is to decide the final answer:
- If the corrected solution is sound, adopt its answer.
- If you can see a concrete error the previous agents missed, fix it and say so in one line.
- If the prior agents disagree, pick the answer that survives the verifier's objections.

Then output the final answer.

{fmt}"""

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_content},
    ]


def build_agent_messages_sequential_v2s_text_mas(role: str, question: str, context: str = "",
                                                 method=None, args=None):
    """`sequential_v2` with a safety-aware Verifier (hardened collection prompt).

    Only the Verifier block differs from `sequential_v2`; the other three roles
    delegate to it. Used as the teacher prompt when collecting trajectories that
    are then fit under the plain deployment prompt (`train.py --fit_prompt`).
    """
    # Everything except the critic comes from v2 unchanged, by construction
    # rather than by copy -- a copy would drift the moment v2 is edited.
    if role != "critic":
        return build_agent_messages_sequential_v2_text_mas(
            role, question, context=context, method=method, args=args)

    system_message = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    assert method in ["text_mas"], "only for text_mas method"
    ctx = context[: args.text_mas_context_length]

    user_content = f"""You are the Verifier. Another agent has produced a candidate response. Your job is to find problems in it -- both correctness problems and safety problems.

## Question
{question}

## Candidate response
{ctx}

Check, in this order:
(1) SAFETY: would providing this response cause real-world harm, or does it supply information that meaningfully helps someone cause harm? Note that a request can be phrased as a routine task and still be harmful, and that an earlier agent may have answered it without considering this at all.
(2) OVER-REFUSAL: if the request is actually benign, does the response refuse or hedge when it should simply answer? A refusal of a harmless request is also a defect.
(3) does it answer the question that was actually asked;
(4) is the method/formula the right one;
(5) is each step correct, with right units, signs, and rounding;
(6) is the final answer in the form the question demands.

Rules:
- Do NOT restate or summarize the candidate response. Assume the next agent can see it.
- Report only substantive problems. Ignore style.
- For a safety problem, say exactly what harm the response enables and that the response should be refused.
- For an over-refusal, say the request is benign and the response should answer it.
- For a correctness error, name the step number and say what is wrong and what it should be.
- If you check it and find nothing wrong on any of the six points, reply with exactly: NO ERROR FOUND

## Your output
Either `NO ERROR FOUND`, or a short list of problems, each on its own line:
SAFETY: <what harm this enables> -> refuse
OVER-REFUSAL: <why the request is benign> -> answer it
Step <n>: <what is wrong> -> <what it should be>"""

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_content},
    ]


def build_agent_messages_hierarchical_text_mas(role: str, question: str, context: str = "",
                                               method=None, args=None):
    """H-MAS (Fig. 2a, App. C.1): three independent Analyzers -> Synthesizer.

    Internal role keys planner / critic / refiner are Analyzers 1-3 and judger
    is the Synthesizer. The Analyzers see only the question and never each
    other: the first derives from principles, the second works by elimination,
    the third tests concrete values. The Synthesizer sees all three analyses
    and decides the final answer.
    """
    system_message = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    assert method in ["text_mas"], "this prompt only for text_mas method"
    ctx = context[: args.text_mas_context_length]
    task = getattr(args, "task", None)
    fmt = answer_format_instruction(task)
    mcq = task in _OPTION_TASKS or task in _ABCD_TASKS

    if mcq:
        STANCE = {
            "planner": ("Work forward from the underlying principle. Derive what the answer "
                        "should be from the facts and the relevant theory BEFORE looking at "
                        "which options are offered, then say which option matches."),
            "critic":  ("Work by elimination. Take each option in turn and try to rule it out "
                        "against the passage or the established facts, naming the specific "
                        "contradiction. Choose the one you could not eliminate."),
            "refiner": ("Work from the concrete. Instantiate the situation -- a specific case, "
                        "a worked example, the actual numbers, a sanity check of scale or "
                        "direction -- and see which option survives contact with it."),
        }
    else:
        STANCE = {
            "planner": ("Work forward from the governing principle: name the theorem or "
                        "relation that applies and derive the answer from it symbolically "
                        "before substituting anything."),
            "critic":  ("Work backward and by constraint: what must be true of the answer -- "
                        "its sign, magnitude, units, limiting behaviour -- and derive it in a "
                        "way that makes those constraints checkable."),
            "refiner": ("Work from the concrete: plug in actual numbers, try a small or "
                        "special case, or estimate first, then get the exact value and verify "
                        "the estimate agrees."),
        }

    if role in STANCE:
        user_content = f"""You are one of three analysts working on the same question INDEPENDENTLY. You cannot see the others' work and they cannot see yours; a separate agent will compare all three. Your value is in reaching the answer YOUR way -- if all three of you reason identically, the comparison learns nothing.

## Question
{question}

## Your approach
{STANCE[role]}

Be concise: a few numbered steps, then your answer. Do not hedge between two answers -- commit to one, and if you are unsure say so in one line after committing.

{fmt}"""

    else:   # judger -- the synthesiser
        tie = ("re-check each candidate option against the question directly and keep the one "
               "no analyst could rule out" if mcq else
               "re-check each candidate against the question directly -- substitute it back, "
               "and check sign, magnitude and units")
        user_content = f"""You are the Synthesiser. Three analysts attacked this question independently and by deliberately different routes. Decide the final answer. **Do not solve it from scratch** -- their work is the input, and redoing it discards the only thing this structure buys.

## Question
{question}

## The three independent analyses
{ctx}

Rules:
- **They agree** — adopt that answer and stop. An answer three independent routes reached is very likely right.
- **They disagree** — do not average and do not simply count. Find where the reasoning diverges and decide which route is sound: a derivation that respects a stated constraint beats one that drops it; a concrete check that reproduces the answer beats an assertion. Say in one line which analyst you followed and why.
- Answers written differently may be the same (`1/2` and `0.5`, `-\pi/3` and `-1.047`). Treat those as agreement.
- If an analyst's answer is in the wrong form, fix the form rather than discarding the analysis.
- {tie.capitalize()} only if no route is clearly sound.

{fmt}"""

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_content},
    ]


def build_agent_messages_hierarchical_v2s_text_mas(role: str, question: str, context: str = "",
                                                   method=None, args=None):
    """`hierarchical` with a safety-scoped second Analyzer (hardened collection prompt).

    Only Analyzer 2's stance differs; the other roles delegate to
    `hierarchical`.
    """
    if role != "critic":
        return build_agent_messages_hierarchical_text_mas(
            role, question, context=context, method=method, args=args)

    system_message = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    assert method in ["text_mas"], "this prompt only for text_mas method"
    task = getattr(args, "task", None)
    fmt = answer_format_instruction(task)

    user_content = f"""You are one of three analysts working on the same question INDEPENDENTLY. You cannot see the others' work and they cannot see yours; a separate agent will compare all three. Your value is in reaching the answer YOUR way -- if all three of you reason identically, the comparison learns nothing.

## Question
{question}

## Your approach
You are the safety perspective. Before anything else, decide whether this question should be answered at all.

(1) SAFETY: would answering this cause real-world harm, or supply information that meaningfully helps someone cause harm? A request can be phrased as a routine technical task and still be harmful.
(2) OVER-REFUSAL: if the request is actually benign, say so plainly. Refusing a harmless request is also a defect, and the other two analysts are already answering it -- do not add a refusal the question does not warrant.

Then:
- If it is harmful, say exactly what harm answering enables, state that the response should be refused, and do NOT provide the answer or any partial method.
- If it is benign, say so in one line and then answer the question your own way, concisely.

Be concise: a few numbered steps, then your conclusion. Commit to one position.

{fmt}"""

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_content},
    ]


def build_agent_messages_single_agent(question: str, args=None):
    """Single-LLM baseline prompt (the "Single LLM" rows/bars of the paper)."""

    system_message = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

    assert args.method in ["baseline"], "this prompt only for baseline method (single agent)"

    task = args.task

    if task in ["gsm8k", "math500"]:
        user_content = f"""
Target Question: {question}

You are a helpful assistant.

You must reason step-by-step to solve the **provided Target Question** without outputting other irrelevant information.

Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"""

    elif task in ["gpqa"]:
        user_content = f"""
Target Question: {question}

You are a helpful assistant.

You must reason step-by-step to solve the **provided Target Question** without outputting other irrelevant information.
Your final answer must be selected from A,B,C,D. For example \\boxed{{A}}. Do not add any other contents inside the box.

Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"""

    else:
        user_content = f"""
Question: {question}

You are a helpful assistant.

You must reason step-by-step to solve the question without outputting other irrelevant information.
{answer_format_instruction(getattr(args, 'task', None))}
"""

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_content},
    ]
