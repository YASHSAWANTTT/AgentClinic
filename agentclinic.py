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
def query_model(model_str, prompt, system_prompt, tries=30, timeout=20.0, image_requested=False, scene=None, max_prompt_len=2**14, clip_prompt=False):
    if model_str not in ["gpt4", "gpt3.5", "gpt4o", 'llama-2-70b-chat', "mixtral-8x7b", "gpt-4o-mini", "llama-3-70b-instruct", "gpt4v", "claude3.5sonnet", "o1-preview"] and "_HF" not in model_str:
        raise Exception("No model by the name {}".format(model_str))
    for _ in range(tries):
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
                answer = response["choices"][0]["message"]["content"]

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
            time.sleep(timeout)
            continue
    raise Exception("Max retries: timeout")

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
        self.agent_hist += question + "\n\n" + answer + "\n\n"
        self.infs += 1
        return answer

    def system_prompt(self) -> str:
        bias_prompt = ""
        if self.bias_present is not None:
            bias_prompt = self.generate_bias()
        base = "You are a doctor named Dr. Agent who only responds in the form of dialogue. You are inspecting a patient who you will ask questions in order to understand their disease. You are only allowed to ask {} questions total before you must make a decision. You have asked {} questions so far. You can request test results using the format \"REQUEST TEST: [test]\". For example, \"REQUEST TEST: Chest_X-Ray\". Your dialogue will only be 1-3 sentences in length. Once you have decided to make a diagnosis please type \"DIAGNOSIS READY: [diagnosis here]\"".format(self.MAX_INFS, self.infs) + ("You may also request medical images related to the disease to be returned with \"REQUEST IMAGES\"." if self.img_request else "")
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
        presentation = "\n\nBelow is all of the information you have. {}. \n\n Remember, you must discover their disease by asking them questions. You are also able to provide exams.".format(self.presentation)
        # Evidence-lock instruction appended only when enabled
        return base + bias_prompt + presentation + intake_section + (("\n\n" + EVIDENCE_LOCK_INSTRUCTION) if self.evidence_lock else "")

    def reset(self) -> None:
        self.agent_hist = ""
        self.presentation = self.scenario.examiner_information()

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
    answer = query_model(moderator_llm, "\nHere is the correct diagnosis: " + correct_diagnosis + "\n Here was the doctor dialogue: " + diagnosis + "\nAre these the same?", "You are responsible for determining if the corrent diagnosis and the doctor diagnosis are the same disease. Please respond only with Yes or No. Nothing else.")
    return answer.lower()

# -----------------
# Main run function
# -----------------
def main(api_key, replicate_api_key, inf_type, doctor_bias, patient_bias, doctor_llm, patient_llm, measurement_llm, moderator_llm, num_scenarios, dataset, img_request, total_inferences, anthropic_api_key=None, evidence_lock=False, guideline_snippets_path="data/guideline_snippets.csv", use_intake_assistant=False, intake_llm="gpt4", intake_turns=3):
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

    for _scenario_id in range(0, min(num_scenarios, scenario_loader.num_scenarios)):
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
                pi_dialogue += "This is the final question. Please provide a diagnosis.\n"

            # Doctor turn
            if inf_type == "human_doctor":
                doctor_dialogue = input("\nQuestion for patient: ")
            else:
                doctor_dialogue = doctor_agent.inference_doctor(pi_dialogue, image_requested=imgs)

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
    parser.add_argument('--intake_assistant_turns', type=int, default=3, help='Maximum number of intake assistant follow-up questions before summarizing.')

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
        intake_turns=args.intake_assistant_turns
    )
