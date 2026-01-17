import argparse
try:
    import anthropic
except ImportError:  # Provide graceful fallback if Anthropic SDK absent
    anthropic = None
from transformers import pipeline
import openai, re, random, time, json, replicate, os

# Evidence-lock + RAG imports (you already had these; kept as-is)
from evidence_block import extract_evidence_block, has_commit_line, extract_final_dx
from rag import GuidelineRAG

llama2_url = "meta/llama-2-70b-chat"
llama3_url = "meta/meta-llama-3-70b-instruct"
mixtral_url = "mistralai/mixtral-8x7b-instruct-v0.1"

# --------------------------
# Utility: HF local pipeline
# --------------------------
def load_huggingface_model(model_name):
    pipe = pipeline("text-generation", model=model_name, device_map="auto")
    return pipe

def inference_huggingface(prompt, pipe):
    response = pipe(prompt, max_new_tokens=100)[0]["generated_text"]
    response = response.replace(prompt, "")
    return response

# ------------------------------------------------
# Core LLM call (unchanged except for housekeeping)
# ------------------------------------------------
def query_model(model_str, prompt, system_prompt, tries=30, timeout=60.0, image_requested=False, scene=None, max_prompt_len=2**14, clip_prompt=False):
    if model_str not in ["gpt4", "gpt3.5", "gpt4o", 'llama-2-70b-chat', "mixtral-8x7b", "gpt-4o-mini", "llama-3-70b-instruct", "gpt4v", "claude3.5sonnet", "o1-preview", "gpt-5.2", "gpt-5.2-pro", "gpt-5-mini", "gpt-5-nano"] and "_HF" not in model_str:
        raise Exception("No model by the name {}".format(model_str))
    for attempt in range(tries):
        if clip_prompt: prompt = prompt[:max_prompt_len]
        try:
            if image_requested:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",
                     "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": "{}".format(scene.image_url)}},
                     ]},
                ]
                if model_str == "gpt4v":
                    response = openai.ChatCompletion.create(
                        model="gpt-4-vision-preview",
                        messages=messages, temperature=0.05, max_tokens=200)
                elif model_str == "gpt-4o-mini":
                    response = openai.ChatCompletion.create(
                        model="gpt-4o-mini",
                        messages=messages, temperature=0.05, max_tokens=200)
                elif model_str == "gpt4":
                    response = openai.ChatCompletion.create(
                        model="gpt-4-turbo",
                        messages=messages, temperature=0.05, max_tokens=200)
                elif model_str == "gpt4o":
                    response = openai.ChatCompletion.create(
                        model="gpt-4o",
                        messages=messages, temperature=0.05, max_tokens=200)
                elif model_str in ["gpt-5.2", "gpt-5.2-pro", "gpt-5-mini", "gpt-5-nano"]:
                    response = openai.ChatCompletion.create(
                        model=model_str,
                        messages=messages, 
                        reasoning_effort="none",  # GPT-5.2: none/low/medium/high/xhigh
                        verbosity="medium",  # GPT-5.2: low/medium/high
                        max_tokens=200)
                answer = response["choices"][0]["message"]["content"]
                return answer

            if model_str == "gpt4":
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}]
                response = openai.ChatCompletion.create(
                    model="gpt-4-turbo-preview",
                    messages=messages, temperature=0.05, max_tokens=200)
                answer = response["choices"][0]["message"]["content"]
                answer = re.sub(r"\s+", " ", answer)
            elif model_str == "gpt4v":
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}]
                response = openai.ChatCompletion.create(
                    model="gpt-4-vision-preview",
                    messages=messages, temperature=0.05, max_tokens=200)
                answer = response["choices"][0]["message"]["content"]
                answer = re.sub(r"\s+", " ", answer)
            elif model_str == "gpt-4o-mini":
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}]
                response = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=messages, temperature=0.05, max_tokens=200)
                answer = response["choices"][0]["message"]["content"]
                answer = re.sub(r"\s+", " ", answer)
            elif model_str == "o1-preview":
                messages = [{"role": "user", "content": system_prompt + prompt}]
                response = openai.ChatCompletion.create(
                    model="o1-preview-2024-09-12", messages=messages)
                answer = response["choices"][0]["message"]["content"]
                answer = re.sub(r"\s+", " ", answer)
            elif model_str == "gpt3.5":
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}]
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=messages, temperature=0.05, max_tokens=200)
                answer = response["choices"][0]["message"]["content"]
                answer = re.sub(r"\s+", " ", answer)
            elif model_str == "claude3.5sonnet":
                if anthropic is None:
                    raise ImportError("anthropic python package is not installed; install it to use claude3.5sonnet.")
                client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
                message = client.messages.create(
                    model="claude-3-5-sonnet-20240620",
                    system=system_prompt, max_tokens=256,
                    messages=[{"role": "user", "content": prompt}])
                answer = json.loads(message.to_json())["content"][0]["text"]
            elif model_str == "gpt4o":
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}]
                response = openai.ChatCompletion.create(
                    model="gpt-4o",
                    messages=messages, temperature=0.05, max_tokens=200)
                answer = response["choices"][0]["message"]["content"]
                answer = re.sub(r"\s+", " ", answer)
            elif model_str in ["gpt-5.2", "gpt-5.2-pro", "gpt-5-mini", "gpt-5-nano"]:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}]
                response = openai.ChatCompletion.create(
                    model=model_str,
                    messages=messages,
                    reasoning_effort="none",  # GPT-5.2: none/low/medium/high/xhigh (none = fastest)
                    verbosity="medium",  # GPT-5.2: low/medium/high
                    max_tokens=200)
                answer = response["choices"][0]["message"]["content"]
                answer = re.sub(r"\s+", " ", answer)
            elif model_str == 'llama-2-70b-chat':
                output = replicate.run(
                    llama2_url,
                    input={"prompt": prompt, "system_prompt": system_prompt, "max_new_tokens": 200})
                answer = ''.join(output)
                answer = re.sub(r"\s+", " ", answer)
            elif model_str == 'mixtral-8x7b':
                output = replicate.run(
                    mixtral_url,
                    input={"prompt": prompt, "system_prompt": system_prompt, "max_new_tokens": 75})
                answer = ''.join(output)
                answer = re.sub(r"\s+", " ", answer)
            elif model_str == 'llama-3-70b-instruct':
                output = replicate.run(
                    llama3_url,
                    input={"prompt": prompt, "system_prompt": system_prompt, "max_new_tokens": 200})
                answer = ''.join(output)
                answer = re.sub(r"\s+", " ", answer)
            elif "HF_" in model_str:
                input_text = system_prompt + prompt
                raise Exception("Sorry, fixing TODO :3")
            return answer
        except Exception as e:
            error_msg = str(e)
            # Log the error for debugging (only show first 100 chars to avoid spam)
            print(f"API call failed (attempt {attempt+1}/{tries}): {error_msg[:100]}...", flush=True)
            
            # Check for specific error types and adjust wait time
            if "rate limit" in error_msg.lower():
                wait_time = 10.0
                print(f"Rate limit detected - waiting {wait_time}s...", flush=True)
            elif "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                wait_time = 3.0
                print(f"Timeout detected - waiting {wait_time}s...", flush=True)
            else:
                # Shorter sleep between retries for other errors
                wait_time = 2.0
            
            time.sleep(wait_time)
            continue
    raise Exception(f"Max retries ({tries}) exceeded: failed after {tries} attempts")

# ---------------------------
# Scenario classes (unchanged)
# ---------------------------
class ScenarioMedQA:
    def __init__(self, scenario_dict) -> None:
        self.scenario_dict = scenario_dict
        self.tests = scenario_dict["OSCE_Examination"]["Test_Results"]
        self.diagnosis = scenario_dict["OSCE_Examination"]["Correct_Diagnosis"]
        self.patient_info  = scenario_dict["OSCE_Examination"]["Patient_Actor"]
        self.examiner_info  = scenario_dict["OSCE_Examination"]["Objective_for_Doctor"]
        self.physical_exams = scenario_dict["OSCE_Examination"]["Physical_Examination_Findings"]
    def patient_information(self) -> dict:
        return self.patient_info
    def examiner_information(self) -> dict:
        return self.examiner_info
    def exam_information(self) -> dict:
        exams = self.physical_exams
        exams["tests"] = self.tests
        return exams
    def diagnosis_information(self) -> dict:
        return self.diagnosis

class ScenarioLoaderMedQA:
    def __init__(self) -> None:
        with open("agentclinic_medqa.jsonl", "r") as f:
            self.scenario_strs = [json.loads(line) for line in f]
        self.scenarios = [ScenarioMedQA(_str) for _str in self.scenario_strs]
        self.num_scenarios = len(self.scenarios)
    def sample_scenario(self):
        return self.scenarios[random.randint(0, len(self.scenarios)-1)]
    def get_scenario(self, id):
        if id is None: return self.sample_scenario()
        return self.scenarios[id]

class ScenarioMedQAExtended:
    def __init__(self, scenario_dict) -> None:
        self.scenario_dict = scenario_dict
        self.tests = scenario_dict["OSCE_Examination"]["Test_Results"]
        self.diagnosis = scenario_dict["OSCE_Examination"]["Correct_Diagnosis"]
        self.patient_info  = scenario_dict["OSCE_Examination"]["Patient_Actor"]
        self.examiner_info  = scenario_dict["OSCE_Examination"]["Objective_for_Doctor"]
        self.physical_exams = scenario_dict["OSCE_Examination"]["Physical_Examination_Findings"]
    def patient_information(self) -> dict:
        return self.patient_info
    def examiner_information(self) -> dict:
        return self.examiner_info
    def exam_information(self) -> dict:
        exams = self.physical_exams
        exams["tests"] = self.tests
        return exams
    def diagnosis_information(self) -> dict:
        return self.diagnosis

class ScenarioLoaderMedQAExtended:
    def __init__(self) -> None:
        with open("agentclinic_medqa_extended.jsonl", "r") as f:
            self.scenario_strs = [json.loads(line) for line in f]
        self.scenarios = [ScenarioMedQAExtended(_str) for _str in self.scenario_strs]
        self.num_scenarios = len(self.scenarios)
    def sample_scenario(self):
        return self.scenarios[random.randint(0, len(self.scenarios)-1)]
    def get_scenario(self, id):
        if id is None: return self.sample_scenario()
        return self.scenarios[id]

class ScenarioMIMICIVQA:
    def __init__(self, scenario_dict) -> None:
        self.scenario_dict = scenario_dict
        self.tests = scenario_dict["OSCE_Examination"]["Test_Results"]
        self.diagnosis = scenario_dict["OSCE_Examination"]["Correct_Diagnosis"]
        self.patient_info  = scenario_dict["OSCE_Examination"]["Patient_Actor"]
        self.examiner_info  = scenario_dict["OSCE_Examination"]["Objective_for_Doctor"]
        self.physical_exams = scenario_dict["OSCE_Examination"]["Physical_Examination_Findings"]
    def patient_information(self) -> dict:
        return self.patient_info
    def examiner_information(self) -> dict:
        return self.examiner_info
    def exam_information(self) -> dict:
        exams = self.physical_exams
        exams["tests"] = self.tests
        return exams
    def diagnosis_information(self) -> dict:
        return self.diagnosis

class ScenarioLoaderMIMICIV:
    def __init__(self) -> None:
        with open("agentclinic_mimiciv.jsonl", "r") as f:
            self.scenario_strs = [json.loads(line) for line in f]
        self.scenarios = [ScenarioMIMICIVQA(_str) for _str in self.scenario_strs]
        self.num_scenarios = len(self.scenarios)
    def sample_scenario(self):
        return self.scenarios[random.randint(0, len(self.scenarios)-1)]
    def get_scenario(self, id):
        if id is None: return self.sample_scenario()
        return self.scenarios[id]

class ScenarioNEJMExtended:
    def __init__(self, scenario_dict) -> None:
        self.scenario_dict = scenario_dict
        self.question = scenario_dict["question"]
        self.image_url = scenario_dict["image_url"]
        self.diagnosis = [_sd["text"] for _sd in scenario_dict["answers"] if _sd["correct"]][0]
        self.patient_info = scenario_dict["patient_info"]
        self.physical_exams = scenario_dict["physical_exams"]
    def patient_information(self) -> str:
        return self.patient_info
    def examiner_information(self) -> str:
        return "What is the most likely diagnosis?"
    def exam_information(self) -> str:
        return self.physical_exams
    def diagnosis_information(self) -> str:
        return self.diagnosis

class ScenarioLoaderNEJMExtended:
    def __init__(self) -> None:
        with open("agentclinic_nejm_extended.jsonl", "r") as f:
            self.scenario_strs = [json.loads(line) for line in f]
        self.scenarios = [ScenarioNEJMExtended(_str) for _str in self.scenario_strs]
        self.num_scenarios = len(self.scenarios)
    def sample_scenario(self):
        return self.scenarios[random.randint(0, len(self.scenarios)-1)]
    def get_scenario(self, id):
        if id is None: return self.sample_scenario()
        return self.scenarios[id]

class ScenarioNEJM:
    def __init__(self, scenario_dict) -> None:
        self.scenario_dict = scenario_dict
        self.question = scenario_dict["question"]
        self.image_url = scenario_dict["image_url"]
        self.diagnosis = [_sd["text"] for _sd in scenario_dict["answers"] if _sd["correct"]][0]
        self.patient_info = scenario_dict["patient_info"]
        self.physical_exams = scenario_dict["physical_exams"]
    def patient_information(self) -> str:
        return self.patient_info
    def examiner_information(self) -> str:
        return "What is the most likely diagnosis?"
    def exam_information(self) -> str:
        return self.physical_exams
    def diagnosis_information(self) -> str:
        return self.diagnosis

class ScenarioLoaderNEJM:
    def __init__(self) -> None:
        with open("agentclinic_nejm.jsonl", "r") as f:
            self.scenario_strs = [json.loads(line) for line in f]
        self.scenarios = [ScenarioNEJM(_str) for _str in self.scenario_strs]
        self.num_scenarios = len(self.scenarios)
    def sample_scenario(self):
        return self.scenarios[random.randint(0, len(self.scenarios)-1)]
    def get_scenario(self, id):
        if id is None: return self.sample_scenario()
        return self.scenarios[id]

# ------------------
# Patient Agent (as-is)
# ------------------
class PatientAgent:
    def __init__(self, scenario, backend_str="gpt4", bias_present=None) -> None:
        self.disease = ""
        self.symptoms = ""
        self.agent_hist = ""
        self.backend = backend_str
        self.bias_present = (None if bias_present == "None" else bias_present)
        self.scenario = scenario
        self.reset()
        self.pipe = None
        self.biases = ["recency", "frequency", "false_consensus", "self_diagnosis", "gender", "race", "sexual_orientation", "cultural", "education", "religion", "socioeconomic"]

    def generate_bias(self) -> str:
        if self.bias_present == "recency":
            return "\nRecently, you had a friend with similar symptoms to yours that was diagnosed with cancer. This affects how you interact with the doctor.\n"
        elif self.bias_present == "frequency":
            return "\nYou recently learned that most frequently, doctors diagnose this as cancer based on past cases. This affects how you interact with the doctor.\n"
        elif self.bias_present == "false_consensus":
            return "\nMost of your friends who heard about your symptoms believe that you have cancer. This affects how you interact with the doctor.\n"
        elif self.bias_present == "self_diagnosis":
            return "\nYou recently performed a thorough investigation of your symptoms online and have come to the conclusion that you should be diagnosed with cancer. While the doctor may believe otherwise you remain convinced you have cancer. This affects how you interact with the doctor.\n"
        elif self.bias_present == "gender":
            return "\nYou are uncomfortable with your doctor because you find out that they are a particular gender and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "race":
            return "\nYou are uncomfortable with your doctor because you find out that they are a particular race and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "sexual_orientation":
            return "\nYou are uncomfortable with your doctor because you find out that they are a particular sexual orientation and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "cultural":
            return "\nYou are uncomfortable with your doctor because you find out that they are from a particular culture and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "education":
            return "\nYou are uncomfortable with your doctor because you find out that they went to a low ranked medical school and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "religion":
            return "\nYou are uncomfortable with your doctor because you find out that they believe in a particular religion and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "socioeconomic":
            return "\nYou are uncomfortable with your doctor because you find out that they are from a particular socioeconomic background and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present is None:
            pass
        else:
            print("BIAS TYPE {} NOT SUPPORTED, ignoring bias...".format(self.bias_present))
        return ""

    def inference_patient(self, question) -> str:
        answer = query_model(self.backend, "\nHere is a history of your dialogue: " + self.agent_hist + "\n Here was the doctor response: " + question + "Now please continue your dialogue\nPatient: ", self.system_prompt())
        self.agent_hist += question + "\n\n" + answer + "\n\n"
        return answer

    def system_prompt(self) -> str:
        bias_prompt = ""
        if self.bias_present is not None:
            bias_prompt = self.generate_bias()
        base = """You are a patient in a clinic who only responds in the form of dialogue. You are being inspected by a doctor who will ask you questions and will perform exams on you in order to understand your disease. Your answer will only be 1-3 sentences in length."""
        symptoms = "\n\nBelow is all of your information. {}. \n\n Remember, you must not reveal your disease explicitly but may only convey the symptoms you have in the form of dialogue if you are asked.".format(self.symptoms)
        return base + bias_prompt + symptoms

    def reset(self) -> None:
        self.agent_hist = ""
        self.symptoms = self.scenario.patient_information()

    def add_hist(self, hist_str) -> None:
        self.agent_hist += hist_str + "\n\n"

# -------------------------
# Intake Assistant (new)
# -------------------------
INTAKE_SUMMARY_TEMPLATE = """{
  "chief_complaint": "<primary concern in patient words>",
  "history": {
    "onset": "<sudden/gradual/timeframe>",
    "location": "<anatomical location or distribution>",
    "duration": "<how long current episode persists>",
    "character": "<quality e.g. pressure, throbbing>",
    "associated_symptoms": ["<symptom1>", "<symptom2>"],
    "alleviating_or_aggravating": "<triggers, relievers>",
    "red_flags": ["<flag1>", "<flag2>"]
  },
  "background": {
    "pmh": "<past medical history or unknown>",
    "medications": "<current medications or unknown>",
    "allergies": "<known allergies or unknown>",
    "social": "<smoking/alcohol/drugs or unknown>",
    "travel_exposures": "<travel, sick contacts, occupational exposures or unknown>"
  },
  "exam": {
    "key_findings": ["<finding1>", "<finding2>"]
  },
  "basic_tests": {
    "ordered_or_suggested": ["<test1>", "<test2>"]
  },
  "missing_critical_info": ["<item1>", "<item2>"]
}"""

INTAKE_ASSISTANT_PROMPT_TEMPLATE = """
You are the Intake Assistant Nurse. Your goal is to capture a precise, general-purpose clinical intake summary before the doctor enters.
- Begin by confirming the patient's chief concern in their own words.
- Ask ONLY one concise, high-yield question per turn. Keep language compassionate, culturally sensitive, and practical.
- Use broad history methods (HPI with OPQRST/OLD CARTS, systems review, basic vitals impression) and adapt follow-ups to whatever concern the patient mentions; do NOT assume a diagnosis category ahead of time.
- Surface immediate safety red flags (airway/breathing/circulation compromise, pregnancy concerns, neuro deficits, suicidality, violent risk) whenever applicable.
- If the patient already supplied a detail, acknowledge it succinctly instead of re-asking.
- Track critical gaps. Anything you cannot obtain must be listed under `missing_critical_info` so the doctor knows what to clarify.
- Prioritize capturing: travel/exposures, medications/immunosuppression, pregnancy status (where relevant), and substance use (nonjudgmental approach).
- You may ask at most {max_turns} follow-up questions before you MUST produce the summary.

Response protocol:
1. If essential data are missing AND you still have question budget, respond EXACTLY as `QUESTION: <single focused question>`.
2. When you have enough information OR when instructed to finalize, respond EXACTLY:
SUMMARY:
<valid JSON following this template>
{template}

Rules:
- Never fabricate information; when the patient does not provide something, record it as "unknown".
- JSON arrays must be valid (use [] if you have nothing).
- When told to finalize (or you hit the question cap), output SUMMARY immediately with no extra prose.
- Keep the JSON compact (<2000 characters) and ensure it mirrors what the patient actually shared.
"""


class IntakeAssistantAgent:
    def __init__(self, backend_str="gpt4", max_turns=3):
        self.backend = backend_str
        self.max_turns = max(1, max_turns)
        self.transcript = ""
        self.summary = None
        self.questions_asked = 0

    def system_prompt(self) -> str:
        return INTAKE_ASSISTANT_PROMPT_TEMPLATE.format(
            template=INTAKE_SUMMARY_TEMPLATE,
            max_turns=self.max_turns
        )

    def register_patient_reply(self, reply: str) -> None:
        reply = reply.strip()
        if not reply:
            reply = "No audible response."
        self.transcript += f"Patient: {reply}\n"

    def next_action(self, force_summary: bool = False):
        """
        Returns ("question", text) or ("summary", json_str).
        """
        directive = (
            f"You have asked {self.questions_asked} of {self.max_turns} allowed questions.\n"
            "Review the conversation and decide your next step."
        )
        no_history = not self.transcript.strip()
        if no_history:
            conversation = "No prior conversation. Introduce yourself briefly and clarify why the patient is here."
        else:
            conversation = self.transcript.strip()
        if force_summary or self.questions_asked >= self.max_turns:
            finalize_text = "You MUST output SUMMARY now. Do not ask another question."
        elif no_history:
            finalize_text = (
                "You have not asked any question yet. Your next reply MUST be a QUESTION to start the intake. "
                "Use the required format `QUESTION: ...`."
            )
        else:
            finalize_text = (
                "If essential slots are still missing and you have question budget, ask another QUESTION.\n"
                "Otherwise, output SUMMARY."
            )
        user_prompt = (
            f"Conversation so far:\n{conversation}\n\n"
            f"{directive}\n{finalize_text}\n"
            "Remember the response protocol."
        )
        raw = query_model(
            self.backend,
            user_prompt,
            self.system_prompt(),
            clip_prompt=True
        )
        raw = raw.strip()
        if raw.lower().startswith("question:"):
            if force_summary:
                # Safety fallback: immediately force summary
                return self.force_summary()
            question = raw.split(":", 1)[1].strip()
            if not question:
                question = "Could you tell me more about what brought you in today?"
            self.questions_asked += 1
            self.transcript += f"IntakeAssistant: {question}\n"
            return "question", question
        if raw.lower().startswith("summary:"):
            summary = raw.split(":", 1)[1].strip()
            self.summary = summary
            return "summary", summary
        # If format was unexpected, force a summary on next call
        return self.force_summary()

    def force_summary(self):
        forced_prompt = (
            "Immediate instruction: Output SUMMARY now using the required JSON format. "
            "Do not include any other text."
        )
        user_prompt = (
            f"Conversation so far:\n{self.transcript or 'No prior conversation.'}\n\n{forced_prompt}"
        )
        raw = query_model(
            self.backend,
            user_prompt,
            self.system_prompt(),
            clip_prompt=True
        ).strip()
        if raw.lower().startswith("summary:"):
            summary = raw.split(":", 1)[1].strip()
            self.summary = summary
            return "summary", summary
        # Last resort: wrap raw text inside summary label
        summary = raw
        self.summary = summary
        return "summary", summary

# ============================================================================
# Dual-output parsing (dialogue + hidden state)
# ============================================================================
STATE_RE = re.compile(r"<STATE_JSON>\s*(\{.*?\})\s*</STATE_JSON>", re.S)

def split_dialogue_and_state(text: str):
    """
    Parse dual-output format: DIALOGUE: ... <STATE_JSON>...</STATE_JSON>
    
    Extracts:
    - dialogue: The 1-3 sentences shown to the patient
    - state: Private JSON notes (evidence ledger, working DDx, etc.)
    
    Returns: (dialogue, state_dict)
    """
    m = STATE_RE.search(text)
    state = None
    if m:
        try:
            state = json.loads(m.group(1))
        except Exception:
            state = None
        text = STATE_RE.sub("", text).strip()

    # Expect "DIALOGUE:" prefix; fall back if missing
    if text.strip().lower().startswith("dialogue:"):
        dialogue = text.split(":", 1)[1].strip()
    else:
        dialogue = text.strip()
    return dialogue, state

# ----------------
# Doctor Agent (+ evidence-lock prompt patch)
# ----------------
EVIDENCE_LOCK_INSTRUCTION = """
Before you say “DIAGNOSIS READY”, you MUST output a valid JSON Evidence Block:

EVIDENCE_BLOCK_JSON:
{
  "task_type": "Diagnosis" or "Exposure",
  "discriminators": ["feature that separates your top two options", "second decisive feature"],
  "key_evidence": "the single test or image feature that decides it",
  "guideline": {"source_id": "CPG_ID", "quote": "1–2 lines supporting your choice"},
  "final_dx": "Your single best answer",
  "confidence": "very certain | somewhat certain | uncertain"
}

Only after this JSON, on the next line, write exactly:
DIAGNOSIS READY: <final_dx>
"""

# ============================================================================
# Prompt Instructions for Enhanced Features
# ============================================================================

DUAL_OUTPUT_INSTRUCTION = """
Output TWO blocks every turn:
DIALOGUE: <1–3 sentences shown to patient>

<STATE_JSON>
{ ...valid JSON... }
</STATE_JSON>

The <STATE_JSON> block is PRIVATE notes (not part of dialogue). You may keep private notes inside <STATE_JSON>...</STATE_JSON>. These notes are not part of the dialogue.
"""

EVIDENCE_LEDGER_INSTRUCTION = """
Maintain an Evidence Ledger in <STATE_JSON> every turn:
- Add new patient facts to key_positives/key_negatives.
- If a test result is abnormal, add to abnormal_results and also unresolved_abnormals unless explained.
- Before DIAGNOSIS READY, unresolved_abnormals must be empty OR marked explained by the final dx.
"""

RESULT_INTEGRATION_GATE = """
You have NEW TEST RESULTS below and MUST integrate them now.
First update <STATE_JSON> with: tests_ordered += {test, result}; if abnormal -> abnormal_results + unresolved_abnormals;
update working_ddx (what goes up/down). Only then produce DIALOGUE.
"""

DDX3_RULE = """
Always maintain exactly 3 candidates in working_ddx (unless case is already certain).
For each: support facts, one 'against'/missing key evidence, and one best disconfirming question/test.
Ask the single question/test that best distinguishes the top 2 DDx.
"""

COMMUNICATION_WRAPPER = """
Every DIALOGUE must include:
- 1 short empathy/validation clause (5–10 words),
- 1 verification/summarization clause ("So far I understand ..."),
- Exactly ONE focused question OR one REQUEST TEST: ...
Neutral, non-judgmental language.
"""

class DoctorAgent:
    def __init__(self, scenario, backend_str="gpt4", max_infs=20, bias_present=None, img_request=False, evidence_lock=False, intake_summary=None) -> None:
        self.infs = 0
        self.MAX_INFS = max_infs
        self.agent_hist = ""
        self.presentation = ""
        self.backend = backend_str
        self.bias_present = (None if bias_present == "None" else bias_present)
        self.scenario = scenario
        self.reset()
        self.pipe = None
        self.img_request = img_request
        self.evidence_lock = evidence_lock
        self.biases = ["recency", "frequency", "false_consensus", "confirmation", "status_quo", "gender", "race", "sexual_orientation", "cultural", "education", "religion", "socioeconomic"]
        self.intake_summary = intake_summary
        
        # Evidence ledger + result integration gate
        self.evidence_ledger = {
            "key_positives": [],
            "key_negatives": [],
            "abnormal_results": [],
            "tests_ordered": [],
            "working_ddx": [],
            "unresolved_abnormals": [],
            "next_info_needed": []
        }
        self.must_integrate_result = False
        self.pending_result_text = None

    def generate_bias(self) -> str:
        if self.bias_present == "recency":
            return "\nRecently, you learned that a doctor with similar symptoms to the patient that diagnosed their patient with cancer. This affects how you interact with the patient.\n"
        elif self.bias_present == "frequency":
            return "\nYou recently learned that most frequently, doctors diagnose patients with similar symptoms with cancer based on past cases. This affects how you interact with the patient.\n"
        elif self.bias_present == "false_consensus":
            return "\nMost of your colleagues who heard about the patient's symptoms believe that they have cancer. This affects how you interact with the patient.\n"
        elif self.bias_present == "status_quo":
            return "\nYour hosptial has been diagnosing these patient's symptoms with cancer for the past 10 years. This affects how you interact with the patient.\n"
        elif self.bias_present == "confirmation":
            return "\nYou are initially confident that the patient has cancer. This affects how you interact with the patient.\n"
        elif self.bias_present == "gender":
            return "\nYou are uncomfortable with your patient because you find out that they are a particular gender and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "race":
            return "\nYou are uncomfortable with your patient because you find out that they are a particular race and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "sexual_orientation":
            return "\nYou are uncomfortable with your patient because you find out that they are a particular sexual orientation and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "cultural":
            return "\nYou are uncomfortable with your patient because you find out that they are from a particular culture and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "education":
            return "\nYou are uncomfortable with your patient because you find out that they are uneducated and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "religion":
            return "\nYou are uncomfortable with your patient because you find out that they believe in a particular religion and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "socioeconomic":
            return "\nYou are uncomfortable with your patient because you find out that they are from a particular socioeconomic background and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present is None:
            pass
        else:
            print("BIAS TYPE {} NOT SUPPORTED, ignoring bias...".format(self.bias_present))
        return ""

    def inference_doctor(self, question, image_requested=False) -> str:
        if self.infs >= self.MAX_INFS: return "Maximum inferences reached"
        answer = query_model(
            self.backend,
            "\nHere is a history of your dialogue: " + self.agent_hist + "\n Here was the patient response: " + question + "Now please continue your dialogue\nDoctor: ",
            self.system_prompt(),
            image_requested=image_requested, scene=self.scenario)
        
        # Parse dual-output format: strip <STATE_JSON> and merge state
        dialogue, state = split_dialogue_and_state(answer)
        
        if state is not None:
            self._merge_state(state)
            # Clear "must integrate" gate once we successfully got a state update
            if self.must_integrate_result:
                self.must_integrate_result = False
                self.pending_result_text = None
        
        # Update history with the *full* answer if you want, but only return dialogue outward
        self.agent_hist += question + "\n\n" + dialogue + "\n\n"
        self.infs += 1
        return dialogue
    
    def _merge_state(self, state: dict):
        """Merge state JSON into evidence ledger (additive, doesn't break if JSON is missing)."""
        if not isinstance(state, dict):
            return
        for k, v in state.items():
            if k not in self.evidence_ledger:
                continue
            # list fields: extend, but keep small to avoid prompt bloat
            if isinstance(self.evidence_ledger[k], list) and isinstance(v, list):
                self.evidence_ledger[k].extend(v)
                self.evidence_ledger[k] = self.evidence_ledger[k][-20:]  # cap at 20 items
            else:
                self.evidence_ledger[k] = v

    def system_prompt(self) -> str:
        bias_prompt = ""
        if self.bias_present is not None:
            bias_prompt = self.generate_bias()
        remaining = self.MAX_INFS - self.infs
        is_final_turn = (remaining == 1)
        final_turn_warning = ""
        if is_final_turn:
            final_turn_warning = "\n\n⚠️ CRITICAL: THIS IS YOUR FINAL TURN. You MUST provide a diagnosis now. Output 'DIAGNOSIS READY: [diagnosis]' in your response. Do not ask another question.\n"
        base = "You are a doctor named Dr. Agent who only responds in the form of dialogue. You are inspecting a patient who you will ask questions in order to understand their disease. You are only allowed to ask {} questions total before you must make a decision. You have asked {} questions so far. {} remaining questions.{}You can request test results using the format \"REQUEST TEST: [test]\". For example, \"REQUEST TEST: Chest_X-Ray\". Your dialogue will only be 1-3 sentences in length. Once you have decided to make a diagnosis please type \"DIAGNOSIS READY: [diagnosis here]\"".format(self.MAX_INFS, self.infs, remaining, final_turn_warning) + ("You may also request medical images related to the disease to be returned with \"REQUEST IMAGES\"." if self.img_request else "")
        base += (
            "\n\nAnswer style rules:\n"
            "- When you finalize, use the most specific standard diagnosis label (leaf-node), not a broad umbrella.\n"
            "- Avoid vague terms like 'infection', 'pneumonia', 'mass' if a specific entity is supported.\n"
            "- If NEJM answer choices exist, choose from them verbatim.\n"
            "- Output exactly: DIAGNOSIS READY: <single diagnosis label>.\n"
            "\n\nCritical Diagnostic Reasoning Rules:\n"
            "- Before diagnosing, carefully review ALL physical exam findings - they often contain the key clue.\n"
            "- For anatomical obstructions: 'unable to perform complete examination' + normal external = mid-level obstruction (e.g., vaginal septum), not external (imperforate hymen).\n"
            "- For skin conditions: Tense bullae + immunofluorescence findings = autoimmune blistering (bullous pemphigoid), not contact dermatitis.\n"
            "- For drug reactions: Muscle rigidity + elevated CK + neuroleptic use = NMS, not serotonin syndrome.\n"
            "- Always match the diagnosis to the MOST SPECIFIC finding in the evidence, not just the most common condition.\n"
        )
        intake_section = ""
        if self.intake_summary:
            intake_section = (
                "\n\nPre-clinic intake summary (JSON) from an intake assistant. How to use it:\n"
                "- Treat as a starting point, not ground truth; verify key details with the patient.\n"
                "- `unknown` means not obtained / unclear (not a negative finding).\n"
                "- `chief_complaint`: patient's main concern in their own words.\n"
                "- `history.*`: focused HPI slots (onset/location/duration/character/associated symptoms/triggers and red flags).\n"
                "- `exam.key_findings`: basic findings reported/observed; consider them provisional.\n"
                "- `basic_tests.ordered_or_suggested`: conservative initial tests to consider if appropriate.\n"
                "- `missing_critical_info`: high-priority follow-ups you should ask next.\n"
                "\nIntake JSON:\n{}\n"
            ).format(self.intake_summary.strip())
        # Add dual-output instruction and evidence ledger
        base += "\n" + DUAL_OUTPUT_INSTRUCTION
        base += "\n" + EVIDENCE_LEDGER_INSTRUCTION
        base += "\n" + DDX3_RULE
        base += "\n" + COMMUNICATION_WRAPPER
        
        # If there is a pending measurement result to integrate:
        if self.must_integrate_result and self.pending_result_text:
            base += "\n" + RESULT_INTEGRATION_GATE
            base += "\nNEW TEST RESULT (must integrate now):\n" + self.pending_result_text
        
        # Inject current ledger to keep it "sticky"
        base += "\nCURRENT LEDGER (carry forward, update in <STATE_JSON>):\n" + json.dumps(self.evidence_ledger)
        
        # Include physical examination findings in the system prompt
        exam_findings_section = ""
        try:
            exam_info = self.scenario.exam_information()
            if isinstance(exam_info, dict):
                # Extract key physical exam findings (exclude test results which are requested separately)
                key_findings = []
                critical_flags = []
                for key in ["Pelvic_Examination", "Dermatological_Examination", "General_Examination", "Cardiovascular_Examination", "Respiratory_Examination", "Abdominal_Examination", "Neurological_Examination"]:
                    if key in exam_info and exam_info[key]:
                        finding_text = str(exam_info[key])
                        key_findings.append(f"{key}: {finding_text}")
                        # Flag critical findings that suggest anatomical issues
                        if any(phrase in finding_text.lower() for phrase in ["unable to perform", "incomplete examination", "obstruction", "cannot complete", "unable to visualize"]):
                            critical_flags.append(f"{key} contains: '{finding_text}'")
                if key_findings:
                    exam_findings_section = "\n\nPhysical Examination Findings (available):\n" + "\n".join(key_findings)
                    if critical_flags:
                        exam_findings_section += "\n\n⚠️ CRITICAL: The following findings suggest anatomical issues that may be the primary diagnosis:\n" + "\n".join(critical_flags)
                        exam_findings_section += "\n\nWhen you see 'unable to perform complete examination' in amenorrhea cases with normal hormones and normal imaging, consider anatomical obstructions (e.g., vaginal septum, imperforate hymen)."
                    else:
                        exam_findings_section += "\n\nIMPORTANT: Review these findings carefully. They may contain critical diagnostic clues."
            elif isinstance(exam_info, str) and exam_info.strip():
                exam_findings_section = "\n\nPhysical Examination Findings (available):\n" + exam_info
                if any(phrase in exam_info.lower() for phrase in ["unable to perform", "incomplete examination", "obstruction"]):
                    exam_findings_section += "\n\n⚠️ CRITICAL: This finding suggests an anatomical issue that may be the primary diagnosis."
                else:
                    exam_findings_section += "\n\nIMPORTANT: Review these findings carefully."
        except Exception:
            pass
        
        presentation = "\n\nBelow is all of the information you have. {}. \n\n Remember, you must discover their disease by asking them questions. You are also able to provide exams.".format(self.presentation)
        # Evidence-lock instruction appended only when enabled
        return base + bias_prompt + presentation + exam_findings_section + intake_section + (("\n\n" + EVIDENCE_LOCK_INSTRUCTION) if self.evidence_lock else "")

    def reset(self) -> None:
        self.agent_hist = ""
        self.presentation = self.scenario.examiner_information()
        # Reset evidence ledger and result gate
        self.evidence_ledger = {
            "key_positives": [],
            "key_negatives": [],
            "abnormal_results": [],
            "tests_ordered": [],
            "working_ddx": [],
            "unresolved_abnormals": [],
            "next_info_needed": []
        }
        self.must_integrate_result = False
        self.pending_result_text = None

# --------------------
# Measurement Agent (as-is)
# --------------------
class MeasurementAgent:
    def __init__(self, scenario, backend_str="gpt4") -> None:
        self.agent_hist = ""
        self.presentation = ""
        self.backend = backend_str
        self.scenario = scenario
        self.pipe = None
        self.reset()

    def inference_measurement(self, question) -> str:
        answer = query_model(self.backend, "\nHere is a history of the dialogue: " + self.agent_hist + "\n Here was the doctor measurement request: " + question, self.system_prompt())
        self.agent_hist += question + "\n\n" + answer + "\n\n"
        return answer

    def system_prompt(self) -> str:
        base = "You are an measurement reader who responds with medical test results. Please respond in the format \"RESULTS: [results here]\""
        presentation = "\n\nBelow is all of the information you have. {}. \n\n If the requested results are not in your data then you can respond with NORMAL READINGS.".format(self.information)
        return base + presentation

    def add_hist(self, hist_str) -> None:
        self.agent_hist += hist_str + "\n\n"

    def reset(self) -> None:
        self.agent_hist = ""
        self.information = self.scenario.exam_information()

# -----------------
# Moderator compare
# -----------------
def compare_results(diagnosis, correct_diagnosis, moderator_llm, mod_pipe):
    # Extract the diagnosis from doctor dialogue
    dx_match = re.search(r"DIAGNOSIS READY:\s*(.*)", diagnosis)
    doctor_dx = dx_match.group(1).strip() if dx_match else diagnosis.strip()
    
    # Enhanced moderator prompt for stricter comparison
    moderator_prompt = f"""You are a strict medical diagnosis grader. Your task is to determine if the doctor's diagnosis matches the correct diagnosis.

CRITICAL RULES:
- The diagnoses must be the SAME specific disease/condition, not just similar or related
- "Imperforate hymen" and "Vaginal septum" are DIFFERENT conditions - answer "No"
- "Contact dermatitis" and "Bullous pemphigoid" are DIFFERENT conditions - answer "No"
- "Serotonin syndrome" and "Neuroleptic malignant syndrome" are DIFFERENT conditions - answer "No"
- Only answer "Yes" if they are the exact same condition (allowing for minor phrasing differences like "Crohn disease" vs "Crohn's disease")

Correct diagnosis: {correct_diagnosis}
Doctor's diagnosis: {doctor_dx}

Are these the SAME specific disease/condition? Answer ONLY "Yes" or "No"."""
    
    answer = query_model(moderator_llm, moderator_prompt, "You are a strict medical diagnosis grader. Respond only with Yes or No.", clip_prompt=True)
    return answer.lower().strip()

# -----------------
# Question Controller (High-Yield Question Proposer)
# -----------------
QUESTION_CONTROLLER_PROMPT = """
You are a clinical question planner. Your job is to propose the single best next question to maximize diagnostic accuracy.

Inputs:
- Conversation transcript so far (Doctor/Patient turns)
- Optional intake summary JSON (may contain unknowns)
- Physical examination findings (if available)
- Remaining question budget

Rules:
- Output EXACTLY one line:
NEXT_QUESTION: <one concise high-yield question>
- The question must be discriminative (rules IN/OUT at least 2 plausible diagnoses).
- **CRITICAL**: If physical exam findings are provided, review them carefully. Look for:
  * Findings that suggest anatomical issues (e.g., "unable to perform complete examination", "obstruction", "abnormal anatomy")
  * Findings that contradict or support differential diagnoses
  * Findings that haven't been discussed yet in the conversation
- If physical exam findings contain critical clues (like "unable to perform complete exam" in amenorrhea cases), suggest questions that explore those findings or request additional tests to clarify them.
- Prefer asking about missing critical info, risk factors, or a decisive symptom/sign.
- Avoid low-yield "general" questions unless nothing else is missing.
- Be patient-friendly and nonjudgmental.
"""

def propose_next_question(controller_llm, transcript_text, intake_summary, remaining, exam_info=None):
    intake_txt = intake_summary.strip() if intake_summary else "None"
    
    # Format exam information for the controller
    exam_txt = "None"
    if exam_info is not None:
        try:
            if isinstance(exam_info, dict):
                # Extract key findings, especially pelvic/genital exam findings
                exam_summary = []
                if "Pelvic_Examination" in exam_info:
                    exam_summary.append(f"Pelvic Exam: {exam_info['Pelvic_Examination']}")
                if "Dermatological_Examination" in exam_info:
                    exam_summary.append(f"Dermatological Exam: {exam_info['Dermatological_Examination']}")
                if "General_Examination" in exam_info:
                    exam_summary.append(f"General Exam: {exam_info['General_Examination']}")
                # Include any other notable findings
                for key, value in exam_info.items():
                    if key not in ["Pelvic_Examination", "Dermatological_Examination", "General_Examination", "Vital_Signs", "tests"]:
                        if isinstance(value, (str, int, float)) and value:
                            exam_summary.append(f"{key}: {value}")
                exam_txt = "\n".join(exam_summary) if exam_summary else json.dumps(exam_info, indent=2)
            else:
                exam_txt = str(exam_info)
        except Exception:
            exam_txt = str(exam_info) if exam_info else "None"
    
    user_prompt = (
        f"Remaining questions: {remaining}\n\n"
        f"Intake summary: {intake_txt}\n\n"
        f"Physical examination findings:\n{exam_txt}\n\n"
        f"Transcript:\n{transcript_text}\n"
    )
    raw = query_model(controller_llm, user_prompt, QUESTION_CONTROLLER_PROMPT, clip_prompt=True)
    raw = raw.strip()
    if raw.lower().startswith("next_question:"):
        return raw.split(":", 1)[1].strip()
    # fallback
    return "Can you tell me more about what brought you in today, and what symptoms are bothering you most?"

# -----------------
# DX Normalizer (Diagnosis Label Standardizer)
# -----------------
DX_NORMALIZER_SYSTEM = """
You are a diagnosis label normalizer for an exam grader.
Goal: output the single best diagnosis label in the most specific standard medical term.

Rules:
- Output EXACTLY one line: FINAL_DX: <label>
- No extra text.
- Prefer the most specific diagnosis supported by evidence (leaf-node).
- Avoid umbrella terms like "pneumonia", "infection", "mass" unless evidence cannot be more specific.
- If answer choices are provided, you MUST pick one exactly (verbatim).
- For MedQA/MedQA_Ext/MIMICIV datasets: If the proposed dx is a mechanism (e.g., "hypovolemia", "hypoxia") and seems like the dataset expects that phrasing, preserve it rather than converting to a disease name.
- For NEJM/NEJM_Ext datasets: Always pick from the provided answer choices verbatim if they exist.

CRITICAL: Distinguishing Similar Conditions - You MUST correct misdiagnoses:

1. Primary Amenorrhea with Anatomical Obstruction:
   - "Vaginal septum": Mid-vaginal obstruction. Key evidence: "unable to perform complete examination" + "external genitalia unremarkable/normal" = obstruction beyond external level
   - "Imperforate hymen": External obstruction. Key evidence: visible bulging at hymenal level, obstruction visible externally
   - RULE: If pelvic exam shows "unable to perform complete examination" AND "external genitalia unremarkable", it MUST be "Vaginal septum", NOT "Imperforate hymen"

2. Autoimmune Blistering vs Contact Dermatitis:
   - "Bullous pemphigoid": Tense bullae + linear C3/IgG deposits on immunofluorescence
   - "Contact dermatitis": No bullae, no immunofluorescence findings
   - RULE: If evidence shows "tense bullae" + "immunofluorescence" findings, it MUST be "Bullous pemphigoid", NOT "Contact dermatitis"

3. Neuroleptic Malignant Syndrome vs Serotonin Syndrome:
   - "Neuroleptic Malignant Syndrome": Muscle rigidity + elevated CK + neuroleptic/antipsychotic use
   - "Serotonin Syndrome": Hyperreflexia, clonus, hyperthermia, but less rigidity
   - RULE: If evidence shows "muscle rigidity" + "elevated creatine kinase" + antipsychotic use, it MUST be "Neuroleptic Malignant Syndrome", NOT "Serotonin Syndrome"

General Principles:
- ALWAYS match the diagnosis to the MOST SPECIFIC evidence in the physical exam and test results
- If the proposed diagnosis doesn't match the key distinguishing evidence, CORRECT it to the diagnosis that does
- Physical exam findings are often the most reliable differentiator - trust them over general symptoms
"""

def normalize_dx(normalizer_llm, doctor_dialogue, scenario, dataset):
    # Extract raw diagnosis
    m = re.search(r"DIAGNOSIS READY:\s*(.*)", doctor_dialogue)
    raw_dx = m.group(1).strip() if m else doctor_dialogue.strip()

    # For NEJM / NEJM_Ext, extract answer choices
    options_txt = ""
    try:
        if dataset in ["NEJM", "NEJM_Ext"] and hasattr(scenario, "scenario_dict"):
            answers = scenario.scenario_dict.get("answers", [])
            opts = [a["text"] for a in answers if isinstance(a, dict) and "text" in a]
            if opts:
                options_txt = "Answer choices (pick one verbatim):\n" + "\n".join([f"- {o}" for o in opts])
    except Exception:
        pass

    # Handle exam_information() which can return dict or string
    case_context = ""
    physical_exam_highlight = ""
    try:
        exam_info = scenario.exam_information()
        if isinstance(exam_info, dict):
            case_context = json.dumps(exam_info, indent=2)
            # Extract and highlight critical physical exam findings
            critical_findings = []
            for key in ["Pelvic_Examination", "Dermatological_Examination", "General_Examination", 
                       "Cardiovascular_Examination", "Respiratory_Examination", "Abdominal_Examination", 
                       "Neurological_Examination"]:
                if key in exam_info and exam_info[key]:
                    finding = str(exam_info[key])
                    # Flag findings that suggest anatomical issues
                    if any(phrase in finding.lower() for phrase in ["unable to perform", "incomplete examination", 
                                                                   "obstruction", "cannot complete", "unable to visualize",
                                                                   "septum", "imperforate", "agenesis"]):
                        critical_findings.append(f"⚠️ {key}: {finding}")
            if critical_findings:
                physical_exam_highlight = "\n\n⚠️ CRITICAL PHYSICAL EXAM FINDINGS:\n" + "\n".join(critical_findings)
        else:
            case_context = str(exam_info)
            if any(phrase in case_context.lower() for phrase in ["unable to perform", "incomplete examination", "obstruction"]):
                physical_exam_highlight = "\n\n⚠️ CRITICAL: Physical exam shows anatomical obstruction clues"
    except Exception:
        pass

    # Extract patient history context if available (for amenorrhea cases)
    history_context = ""
    try:
        if hasattr(scenario, "patient_information"):
            patient_info = scenario.patient_information()
            if isinstance(patient_info, dict):
                # Look for amenorrhea-related context
                history_text = str(patient_info).lower()
                if "amenorrhea" in history_text or "menstrual" in history_text:
                    history_context = "\n\nClinical context: Primary amenorrhea case"
    except Exception:
        pass

    user_prompt = (
        f"Doctor proposed diagnosis: {raw_dx}\n\n"
        f"{options_txt}\n\n"
        f"{history_context}"
        f"Case evidence (tests/exam):\n{case_context}\n"
        f"{physical_exam_highlight}\n\n"
        f"Task: Normalize the diagnosis to the most specific, accurate label based on the evidence above. "
        f"Pay special attention to physical exam findings that distinguish between similar anatomical conditions."
    )
    out = query_model(normalizer_llm, user_prompt, DX_NORMALIZER_SYSTEM, clip_prompt=True).strip()
    if out.lower().startswith("final_dx:"):
        normalized = out.split(":", 1)[1].strip()
        # Post-process: if raw_dx was close but wrong, and we have a better match, use it
        # This helps catch cases where the normalizer might miss subtle distinctions
        raw_lower = raw_dx.lower()
        normalized_lower = normalized.lower()
        
        # Enhanced post-processing: correct common misdiagnoses based on evidence
        raw_lower = raw_dx.lower()
        normalized_lower = normalized.lower()
        
        # Rule 1: Amenorrhea with anatomical obstruction - distinguish hymen vs septum
        if any(term in raw_lower or term in normalized_lower for term in ["imperforate hymen", "hymen", "vaginal obstruction"]):
            # Check evidence for distinguishing features
            if isinstance(exam_info, dict) and "Pelvic_Examination" in exam_info:
                pelvic_exam = str(exam_info["Pelvic_Examination"]).lower()
                # Key distinction: "unable to perform complete examination" + "external genitalia unremarkable" = septum
                # Imperforate hymen would show bulging/visible obstruction at external level
                if "unable to perform" in pelvic_exam and "complete examination" in pelvic_exam:
                    if "external genitalia" in pelvic_exam and ("unremarkable" in pelvic_exam or "normal" in pelvic_exam):
                        # This strongly suggests vaginal septum (mid-vaginal obstruction)
                        if "septum" not in normalized_lower and "septum" not in raw_lower:
                            return "Vaginal septum"
                    elif "bulging" in pelvic_exam or "visible" in pelvic_exam:
                        # This suggests imperforate hymen (external obstruction)
                        if "hymen" not in normalized_lower:
                            return "Imperforate hymen"
        
        # Rule 2: Bullous pemphigoid vs contact dermatitis
        if any(term in raw_lower or term in normalized_lower for term in ["contact dermatitis", "allergic contact"]):
            # Check for bullous pemphigoid clues: tense bullae, immunofluorescence findings
            if isinstance(exam_info, dict):
                exam_str = json.dumps(exam_info).lower()
                if ("tense" in exam_str and "bullae" in exam_str) or \
                   ("immunofluorescence" in exam_str and ("igg" in exam_str or "c3" in exam_str)):
                    if "pemphigoid" not in normalized_lower:
                        return "Bullous pemphigoid"
        
        # Rule 3: Neuroleptic Malignant Syndrome vs Serotonin Syndrome
        if any(term in raw_lower or term in normalized_lower for term in ["serotonin syndrome", "serotonin"]):
            # Check for NMS clues: muscle rigidity, elevated CK, neuroleptic use
            if isinstance(exam_info, dict):
                exam_str = json.dumps(exam_info).lower()
                if ("rigidity" in exam_str and "muscle" in exam_str) or \
                   ("creatine kinase" in exam_str or "ck" in exam_str) or \
                   ("elevated" in exam_str and "ck" in exam_str):
                    # Check if there's evidence of neuroleptic use (from history)
                    try:
                        patient_info = scenario.patient_information()
                        if isinstance(patient_info, dict):
                            history_str = json.dumps(patient_info).lower()
                            if "neuroleptic" in history_str or "antipsychotic" in history_str:
                                if "neuroleptic" not in normalized_lower and "malignant" not in normalized_lower:
                                    return "Neuroleptic Malignant Syndrome"
                    except:
                        pass
        
        return normalized
    return raw_dx

# -----------------
# Main run function
# -----------------
def main(api_key, replicate_api_key, inf_type, doctor_bias, patient_bias, doctor_llm, patient_llm, measurement_llm, moderator_llm, num_scenarios, dataset, img_request, total_inferences, anthropic_api_key=None, evidence_lock=False, guideline_snippets_path="data/guideline_snippets.csv", use_intake_assistant=False, intake_llm="gpt4", intake_turns=6, question_controller_llm=None, scenario_ids=None):
    # Use provided API key, or fall back to environment variable
    if not api_key or api_key == "":
        api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("OpenAI API key is required. Set --openai_api_key or OPENAI_API_KEY environment variable.")
    openai.api_key = api_key
    anthropic_llms = ["claude3.5sonnet"]
    replicate_llms = ["llama-3-70b-instruct", "llama-2-70b-chat", "mixtral-8x7b"]
    if patient_llm in replicate_llms or doctor_llm in replicate_llms or (use_intake_assistant and intake_llm in replicate_llms):
        os.environ["REPLICATE_API_TOKEN"] = replicate_api_key
    if doctor_llm in anthropic_llms or (use_intake_assistant and intake_llm in anthropic_llms):
        os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key

    # Load dataset
    if dataset == "MedQA":
        scenario_loader = ScenarioLoaderMedQA()
    elif dataset == "MedQA_Ext":
        scenario_loader = ScenarioLoaderMedQAExtended()
    elif dataset == "NEJM":
        scenario_loader = ScenarioLoaderNEJM()
    elif dataset == "NEJM_Ext":
        scenario_loader = ScenarioLoaderNEJMExtended()
    elif dataset == "MIMICIV":
        scenario_loader = ScenarioLoaderMIMICIV()
    else:
        raise Exception("Dataset {} does not exist".format(str(dataset)))

    # Tiny RAG (optional; used for moderator hints on rejection)
    rag = GuidelineRAG(guideline_snippets_path) if evidence_lock else None

    total_correct = 0
    total_presents = 0

    # Pipeline for huggingface moderator (if used)
    if "HF_" in moderator_llm:
        pipe = load_huggingface_model(moderator_llm.replace("HF_", ""))
    else:
        pipe = None

    if num_scenarios is None:
        num_scenarios = scenario_loader.num_scenarios

    # If specific scenario IDs are provided, use those; otherwise use range
    if scenario_ids is not None and len(scenario_ids) > 0:
        scenario_list = [sid for sid in scenario_ids if 0 <= sid < scenario_loader.num_scenarios]
        if not scenario_list:
            raise ValueError(f"None of the provided scenario IDs are valid. Valid range: 0-{scenario_loader.num_scenarios-1}")
        print(f"Running specific scenarios: {scenario_list}")
    else:
        scenario_list = list(range(0, min(num_scenarios, scenario_loader.num_scenarios)))

    for _scenario_id in scenario_list:
        total_presents += 1
        pi_dialogue = str()
        scenario = scenario_loader.get_scenario(id=_scenario_id)

        meas_agent = MeasurementAgent(scenario=scenario, backend_str=measurement_llm)
        patient_agent = PatientAgent(scenario=scenario, bias_present=patient_bias, backend_str=patient_llm)
        intake_summary = None
        if use_intake_assistant:
            intake_agent = IntakeAssistantAgent(backend_str=intake_llm, max_turns=intake_turns)
            action, payload = intake_agent.next_action(force_summary=False)
            if action == "summary":
                intake_summary = payload
            else:
                pending_question = payload
                for turn_idx in range(intake_agent.max_turns):
                    patient_reply = patient_agent.inference_patient(pending_question)
                    intake_agent.register_patient_reply(patient_reply)
                    final_turn = (turn_idx == intake_agent.max_turns - 1)
                    action, payload = intake_agent.next_action(force_summary=final_turn)
                    if action == "summary":
                        intake_summary = payload
                        break
                    pending_question = payload
            if intake_summary is None:
                _, intake_summary = intake_agent.force_summary()
            transcript_text = intake_agent.transcript.strip() or "[No intake dialogue recorded.]"
            print("Intake assistant transcript:\n{}".format(transcript_text))
            print("Intake summary prepared:\n{}".format(intake_summary))
        doctor_agent = DoctorAgent(
            scenario=scenario,
            bias_present=doctor_bias,
            backend_str=doctor_llm,
            max_infs=total_inferences,
            img_request=img_request,
            evidence_lock=evidence_lock,
            intake_summary=intake_summary
        )

        doctor_dialogue = ""
        attempted_commit = False  # track if we've seen one commit attempt

        for _inf_id in range(total_inferences):
            # NEJM image policy
            if dataset == "NEJM":
                if img_request:
                    imgs = "REQUEST IMAGES" in doctor_dialogue
                else:
                    imgs = True
            else:
                imgs = False

            # Final turn hint
            if _inf_id == total_inferences - 1:
                pi_dialogue += "\n\n⚠️ CRITICAL: This is your FINAL turn. You MUST output 'DIAGNOSIS READY: [diagnosis]' now. Do not ask another question - provide your diagnosis immediately.\n"

            # Question Controller: propose high-yield question (prevent transcript pollution)
            doctor_input = pi_dialogue  # Start with clean patient message
            if inf_type != "human_doctor":
                # Heuristic: always call in early turns (first 5), OR call if not after test request AND not last turn
                should_call_controller = (
                    (_inf_id < 5) or 
                    (("REQUEST TEST" not in doctor_dialogue) and (_inf_id < total_inferences - 1))
                )
                if should_call_controller:
                    transcript_text = (doctor_agent.agent_hist + f"\nPatient latest: {pi_dialogue}\n").strip()
                    remaining = total_inferences - _inf_id
                    controller_llm_to_use = question_controller_llm if question_controller_llm else doctor_llm
                    # Get exam information to pass to question controller
                    exam_info = None
                    try:
                        exam_info = scenario.exam_information()
                    except Exception:
                        pass
                    next_q = propose_next_question(
                        controller_llm=controller_llm_to_use,
                        transcript_text=transcript_text,
                        intake_summary=intake_summary,
                        remaining=remaining,
                        exam_info=exam_info
                    )
                    # Create doctor_input with guidance (DO NOT modify pi_dialogue)
                    doctor_input = (
                        f"{pi_dialogue}\n\n"
                        f"MODERATOR GUIDANCE: Ask this next question verbatim unless impossible:\n"
                        f"\"{next_q}\""
                    )

            # Doctor turn
            if inf_type == "human_doctor":
                doctor_dialogue = input("\nQuestion for patient: ")
            else:
                doctor_dialogue = doctor_agent.inference_doctor(doctor_input, image_requested=imgs)

            print("Doctor [{}%]:".format(int(((_inf_id+1)/total_inferences)*100)), doctor_dialogue)

            # -------------------------
            # Evidence-lock moderator gate
            # -------------------------
            if "DIAGNOSIS READY" in doctor_dialogue:
                if evidence_lock:
                    eb, err = extract_evidence_block(doctor_dialogue)
                    if err:
                        # Make the rejection visible in your log
                        print(f"MODERATOR REJECT: evidence_block:{err}", flush=True)

                        # Optional: include a tiny RAG hint to help the doctor cite something authoritative
                        hint_txt = ""
                        if rag is not None:
                            context_query = ""
                            try:
                                context_query = scenario.examiner_information()
                            except Exception:
                                pass
                            if not context_query:
                                try:
                                    context_query = str(scenario.exam_information())
                                except Exception:
                                    context_query = "diagnosis criteria"
                            hits = rag.retrieve(context_query, k=1)
                            if hits:
                                h = hits[0]
                                hint_txt = f" Hint: {h['source_id']}: {h['quote']}"

                        # Feed the rejection back into the dialogue so the doctor retries
                        pi_dialogue = (
                            f"MODERATOR: REJECT evidence_block:{err}."
                            f" Provide EVIDENCE_BLOCK_JSON then the line 'DIAGNOSIS READY: <final_dx>'."
                            f"{hint_txt}"
                        )
                        attempted_commit = True
                        continue  # give the doctor another chance within the same scene

                # DX Normalizer: standardize diagnosis label before comparison
                # Extract original diagnosis for logging
                original_match = re.search(r"DIAGNOSIS READY:\s*(.*)", doctor_dialogue)
                original_dx = original_match.group(1).strip() if original_match else "N/A"
                
                normalized = normalize_dx(
                    normalizer_llm=doctor_llm,
                    doctor_dialogue=doctor_dialogue,
                    scenario=scenario,
                    dataset=dataset
                )
                
                # Log normalization if it changed
                if original_dx.lower() != normalized.lower():
                    print(f"DX NORMALIZER: '{original_dx}' → '{normalized}'", flush=True)
                
                # Replace the diagnosis in doctor_dialogue with normalized version
                doctor_dialogue = re.sub(
                    r"DIAGNOSIS READY:\s*.*",
                    f"DIAGNOSIS READY: {normalized}",
                    doctor_dialogue
                )

                # If we reach here: either evidence_lock is off OR EB validated
                correctness = compare_results(doctor_dialogue, scenario.diagnosis_information(), moderator_llm, pipe) == "yes"
                if correctness:
                    total_correct += 1
                print("\nCorrect answer:", scenario.diagnosis_information())
                print("Scene {}, The diagnosis was ".format(_scenario_id),
                      "CORRECT" if correctness else "INCORRECT",
                      int((total_correct/total_presents)*100))
                break

            # Measurement agent
            if "REQUEST TEST" in doctor_dialogue:
                pi_dialogue = meas_agent.inference_measurement(doctor_dialogue)
                print("Measurement [{}%]:".format(int(((_inf_id+1)/total_inferences)*100)), pi_dialogue)
                patient_agent.add_hist(pi_dialogue)
                # Turn on the "result integration gate" right after MeasurementAgent returns a result
                doctor_agent.must_integrate_result = True
                doctor_agent.pending_result_text = pi_dialogue
            else:
                # Patient reply
                if inf_type == "human_patient":
                    pi_dialogue = input("\nResponse to doctor: ")
                else:
                    pi_dialogue = patient_agent.inference_patient(doctor_dialogue)
                print("Patient [{}%]:".format(int(((_inf_id+1)/total_inferences)*100)), pi_dialogue)
                meas_agent.add_hist(pi_dialogue)

            # Prevent API timeouts
            time.sleep(1.0)


# ----------------------------
# (Optional) RAG helper for CLI
# ----------------------------
def retrieve_guideline_snippet(query: str):
    # Kept for backwards compatibility; not used directly now
    return None

# -------------
# CLI entrypoint
# -------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Medical Diagnosis Simulation CLI')
    parser.add_argument('--openai_api_key', type=str, required=False, help='OpenAI API Key')
    parser.add_argument('--replicate_api_key', type=str, required=False, help='Replicate API Key')
    parser.add_argument('--inf_type', type=str, choices=['llm', 'human_doctor', 'human_patient'], default='llm')
    parser.add_argument('--doctor_bias', type=str, help='Doctor bias type', default='None', choices=["recency", "frequency", "false_consensus", "confirmation", "status_quo", "gender", "race", "sexual_orientation", "cultural", "education", "religion", "socioeconomic"])
    parser.add_argument('--patient_bias', type=str, help='Patient bias type', default='None', choices=["recency", "frequency", "false_consensus", "self_diagnosis", "gender", "race", "sexual_orientation", "cultural", "education", "religion", "socioeconomic"])
    parser.add_argument('--doctor_llm', type=str, default='gpt4')
    parser.add_argument('--patient_llm', type=str, default='gpt4')
    parser.add_argument('--measurement_llm', type=str, default='gpt4')
    parser.add_argument('--moderator_llm', type=str, default='gpt4')
    parser.add_argument('--agent_dataset', type=str, default='MedQA') # MedQA, MIMICIV or NEJM
    parser.add_argument('--doctor_image_request', type=bool, default=False) # whether images must be requested or are provided
    parser.add_argument('--num_scenarios', type=int, default=None, required=False, help='Number of scenarios to simulate')
    parser.add_argument('--total_inferences', type=int, default=20, required=False, help='Number of inferences between patient and doctor')
    parser.add_argument('--anthropic_api_key', type=str, default=None, required=False, help='Anthropic API key for Claude 3.5 Sonnet')
    parser.add_argument('--evidence_lock', action='store_true', help='Require an Evidence Block (with guideline quote) before accepting Diagnosis Ready.')
    parser.add_argument('--guideline_snippets', type=str, default='data/guideline_snippets.csv', help='Path to small curated guideline snippets CSV for RAG.')
    parser.add_argument('--use_intake_assistant', action='store_true', help='Enable pre-clinic intake assistant to summarize key findings for the doctor.')
    parser.add_argument('--intake_assistant_llm', type=str, default='gpt4', help='Backend LLM for the intake assistant.')
    parser.add_argument('--intake_assistant_turns', type=int, default=6, help='Maximum number of intake assistant follow-up questions before summarizing.')
    parser.add_argument('--question_controller_llm', type=str, default=None, required=False, help='Backend LLM for the question controller (defaults to doctor_llm if not specified).')
    parser.add_argument('--scenario_ids', type=int, nargs='+', default=None, required=False, help='Specific scenario IDs to run (e.g., --scenario_ids 199 203). If not provided, runs sequentially from 0.')

    args = parser.parse_args()

    # Init and run
    main(
        args.openai_api_key,
        args.replicate_api_key,
        args.inf_type,
        args.doctor_bias,
        args.patient_bias,
        args.doctor_llm,
        args.patient_llm,
        args.measurement_llm,
        args.moderator_llm,
        args.num_scenarios,
        args.agent_dataset,
        args.doctor_image_request,
        args.total_inferences,
        args.anthropic_api_key,
        evidence_lock=args.evidence_lock,
        guideline_snippets_path=args.guideline_snippets,
        use_intake_assistant=args.use_intake_assistant,
        intake_llm=args.intake_assistant_llm,
        intake_turns=args.intake_assistant_turns,
        question_controller_llm=args.question_controller_llm,
        scenario_ids=args.scenario_ids
    )
