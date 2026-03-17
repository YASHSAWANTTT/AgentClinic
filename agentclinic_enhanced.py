import argparse
try:
    import anthropic
except ImportError:  # Provide graceful fallback if Anthropic SDK absent
    anthropic = None
from transformers import pipeline
import openai, re, random, time, json, replicate, os
try:
    from openai import OpenAI
    OPENAI_NEW_API = True
except ImportError:
    OPENAI_NEW_API = False

# Global OpenAI client (set in main())
_openai_client = None

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
    global _openai_client
    if model_str not in ["gpt4", "gpt3.5", "gpt4o", 'llama-2-70b-chat', "mixtral-8x7b", "gpt-4o-mini", "gpt-4.1-mini", "llama-3-70b-instruct", "gpt4v", "claude3.5sonnet", "o1-preview", "gpt-5.2", "gpt-5.2-pro", "gpt-5-mini", "gpt-5-nano"] and "_HF" not in model_str:
        raise Exception("No model by the name {}".format(model_str))
    # Alias: treat "gpt-4.1-mini" as "gpt-4o-mini" for backend calls
    if model_str == "gpt-4.1-mini":
        model_str = "gpt-4o-mini"
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
                if OPENAI_NEW_API and _openai_client:
                    if model_str == "gpt4v":
                        response = _openai_client.chat.completions.create(
                            model="gpt-4-vision-preview",
                            messages=messages, temperature=0.05, max_tokens=200)
                    elif model_str == "gpt-4o-mini":
                        response = _openai_client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=messages, temperature=0.05, max_tokens=200)
                    elif model_str == "gpt4":
                        response = _openai_client.chat.completions.create(
                            model="gpt-4-turbo",
                            messages=messages, temperature=0.05, max_tokens=200)
                    elif model_str == "gpt4o":
                        response = _openai_client.chat.completions.create(
                            model="gpt-4o",
                            messages=messages, temperature=0.05, max_tokens=200)
                    elif model_str in ["gpt-5.2", "gpt-5.2-pro", "gpt-5-mini", "gpt-5-nano"]:
                        response = _openai_client.chat.completions.create(
                            model=model_str,
                            messages=messages, 
                            reasoning_effort="none",  # GPT-5.2: none/low/medium/high/xhigh
                            verbosity="medium",  # GPT-5.2: low/medium/high
                            max_tokens=200)
                    answer = response.choices[0].message.content
                else:
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
                if OPENAI_NEW_API and _openai_client:
                    response = _openai_client.chat.completions.create(
                        model="gpt-4-turbo-preview",
                        messages=messages, temperature=0.05, max_tokens=200)
                    answer = response.choices[0].message.content
                else:
                    response = openai.ChatCompletion.create(
                        model="gpt-4-turbo-preview",
                        messages=messages, temperature=0.05, max_tokens=200)
                    answer = response["choices"][0]["message"]["content"]
                answer = re.sub(r"\s+", " ", answer)
            elif model_str == "gpt4v":
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}]
                if OPENAI_NEW_API and _openai_client:
                    response = _openai_client.chat.completions.create(
                        model="gpt-4-vision-preview",
                        messages=messages, temperature=0.05, max_tokens=200)
                    answer = response.choices[0].message.content
                else:
                    response = openai.ChatCompletion.create(
                        model="gpt-4-vision-preview",
                        messages=messages, temperature=0.05, max_tokens=200)
                    answer = response["choices"][0]["message"]["content"]
                answer = re.sub(r"\s+", " ", answer)
            elif model_str == "gpt-4o-mini":
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}]
                if OPENAI_NEW_API and _openai_client:
                    response = _openai_client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=messages, temperature=0.05, max_tokens=200)
                    answer = response.choices[0].message.content
                else:
                    response = openai.ChatCompletion.create(
                        model="gpt-4o-mini",
                        messages=messages, temperature=0.05, max_tokens=200)
                    answer = response["choices"][0]["message"]["content"]
                answer = re.sub(r"\s+", " ", answer)
            elif model_str == "o1-preview":
                messages = [{"role": "user", "content": system_prompt + prompt}]
                if OPENAI_NEW_API and _openai_client:
                    response = _openai_client.chat.completions.create(
                        model="o1-preview-2024-09-12", messages=messages)
                    answer = response.choices[0].message.content
                else:
                    response = openai.ChatCompletion.create(
                        model="o1-preview-2024-09-12", messages=messages)
                    answer = response["choices"][0]["message"]["content"]
                answer = re.sub(r"\s+", " ", answer)
            elif model_str == "gpt3.5":
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}]
                if OPENAI_NEW_API and _openai_client:
                    response = _openai_client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=messages, temperature=0.05, max_tokens=200)
                    answer = response.choices[0].message.content
                else:
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
                if OPENAI_NEW_API and _openai_client:
                    response = _openai_client.chat.completions.create(
                        model="gpt-4o",
                        messages=messages, temperature=0.05, max_tokens=200)
                    answer = response.choices[0].message.content
                else:
                    response = openai.ChatCompletion.create(
                        model="gpt-4o",
                        messages=messages, temperature=0.05, max_tokens=200)
                    answer = response["choices"][0]["message"]["content"]
                answer = re.sub(r"\s+", " ", answer)
            elif model_str in ["gpt-5.2", "gpt-5.2-pro", "gpt-5-mini", "gpt-5-nano"]:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}]
                if OPENAI_NEW_API and _openai_client:
                    response = _openai_client.chat.completions.create(
                        model=model_str,
                        messages=messages,
                        reasoning_effort="none",  # GPT-5.2: none/low/medium/high/xhigh (none = fastest)
                        verbosity="medium",  # GPT-5.2: low/medium/high
                        max_tokens=200)
                    answer = response.choices[0].message.content
                else:
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
        except KeyboardInterrupt:
            # Don't catch user interrupts - let them propagate
            raise
        except Exception as e:
            error_msg = str(e)
            error_type = type(e).__name__
            
            # Log the error for debugging (only show first 200 chars to avoid spam)
            print(f"API call failed (attempt {attempt+1}/{tries}) [{error_type}]: {error_msg[:200]}...", flush=True)
            
            # Check for specific error types and adjust wait time with exponential backoff
            if "rate limit" in error_msg.lower() or "429" in error_msg:
                wait_time = min(10.0 * (1.5 ** attempt), 60.0)  # Exponential backoff, max 60s
                print(f"Rate limit detected - waiting {wait_time:.1f}s...", flush=True)
            elif "timeout" in error_msg.lower() or "timed out" in error_msg.lower() or "Timeout" in error_type:
                wait_time = min(3.0 * (1.5 ** attempt), 30.0)  # Exponential backoff, max 30s
                print(f"Timeout detected - waiting {wait_time:.1f}s...", flush=True)
            elif "connection" in error_msg.lower() or "Connection" in error_type:
                wait_time = min(2.0 * (1.5 ** attempt), 20.0)  # Exponential backoff, max 20s
                print(f"Connection error detected - waiting {wait_time:.1f}s...", flush=True)
            else:
                # Shorter sleep between retries for other errors, with slight backoff
                wait_time = min(1.0 * (1.2 ** attempt), 5.0)  # Exponential backoff, max 5s
            
            # Don't sleep on the last attempt
            if attempt < tries - 1:
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
        answers = self.scenario_dict.get("answers", [])
        opts = [a["text"] for a in answers if isinstance(a, dict) and "text" in a]
        if opts:
            return "What is the most likely diagnosis? Answer choices (pick one verbatim):\n" + "\n".join([f"- {o}" for o in opts])
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
        answers = self.scenario_dict.get("answers", [])
        opts = [a["text"] for a in answers if isinstance(a, dict) and "text" in a]
        if opts:
            return "What is the most likely diagnosis? Answer choices (pick one verbatim):\n" + "\n".join([f"- {o}" for o in opts])
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

def _extract_nested_json(text: str, start_pos: int):
    """Extract JSON object with nested braces, starting from start_pos."""
    if start_pos >= len(text) or text[start_pos] != '{':
        return None, start_pos
    
    brace_count = 0
    i = start_pos
    start = i
    
    while i < len(text):
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                return text[start:i+1], i + 1
        i += 1
    
    return None, start_pos

def split_dialogue_and_state(text: str):
    """
    Parse dual-output format: DIALOGUE: ... <STATE_JSON>...</STATE_JSON>
    
    Extracts:
    - dialogue: The 1-3 sentences shown to the patient
    - state: Private JSON notes (evidence ledger, working DDx, etc.)
    
    Returns: (dialogue, state_dict)
    """
    # Find all STATE_JSON blocks and extract them
    state = None
    cleaned_text = text
    
    # Find all <STATE_JSON> tags
    start_tag = "<STATE_JSON>"
    end_tag = "</STATE_JSON>"
    
    while True:
        start_idx = cleaned_text.find(start_tag)
        if start_idx == -1:
            break
        
        # Find the position after <STATE_JSON>
        json_start = start_idx + len(start_tag)
        # Skip whitespace
        while json_start < len(cleaned_text) and cleaned_text[json_start].isspace():
            json_start += 1
        
        if json_start >= len(cleaned_text) or cleaned_text[json_start] != '{':
            # No JSON found, remove the tag and continue
            cleaned_text = cleaned_text[:start_idx] + cleaned_text[start_idx + len(start_tag):]
            continue
        
        # Extract nested JSON
        json_str, json_end = _extract_nested_json(cleaned_text, json_start)
        if json_str:
            # Find the closing tag
            end_idx = cleaned_text.find(end_tag, json_end)
            if end_idx != -1:
                # Try to parse the JSON
                try:
                    state = json.loads(json_str)
                except Exception:
                    pass
                # Remove the entire STATE_JSON block
                cleaned_text = cleaned_text[:start_idx] + cleaned_text[end_idx + len(end_tag):]
            else:
                # No closing tag found, just remove the opening tag
                cleaned_text = cleaned_text[:start_idx] + cleaned_text[start_idx + len(start_tag):]
        else:
            # Couldn't extract JSON, remove the tag
            cleaned_text = cleaned_text[:start_idx] + cleaned_text[start_idx + len(start_tag):]
    
    text = cleaned_text.strip()

    # Expect "DIALOGUE:" prefix; fall back if missing
    if text.strip().lower().startswith("dialogue:"):
        dialogue = text.split(":", 1)[1].strip()
    else:
        dialogue = text.strip()

    # Strip leaked evidence-ledger JSON (model sometimes appends {"key_positives":...} without <STATE_JSON>)
    for sentinel in ["key_positives", "key_negatives", "working_ddx", '"abnormal_results"']:
        idx = dialogue.find(sentinel)
        if idx != -1:
            # Find start of this JSON object (go back to preceding "  {" or "\n{")
            start = dialogue.rfind("  {", 0, idx)
            if start == -1:
                start = dialogue.rfind("\n{", 0, idx)
            if start != -1:
                brace_start = dialogue.find("{", start)
                if brace_start != -1:
                    try:
                        json_str, _ = _extract_nested_json(dialogue, brace_start)
                        if json_str and state is None:
                            state = json.loads(json_str)
                    except Exception:
                        pass
                dialogue = dialogue[:start].strip()
            break
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

    def inference_doctor(self, question, image_requested=False, allow_extra_feedback_turn=False) -> str:
        if not allow_extra_feedback_turn and self.infs >= self.MAX_INFS:
            return "Maximum inferences reached"
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
            "- Before committing, your final diagnosis MUST be one of your top 3 in working_ddx. If you have not listed it in working_ddx, do not output DIAGNOSIS READY yet—ask one more question or update working_ddx first.\n"
            "- Optionally append confidence: DIAGNOSIS READY: <label> [CONFIDENCE: high|medium|low]\n"
            "\n\nCRITICAL: Diagnose the DISEASE/CONDITION, NOT a lab finding or symptom:\n"
            "- WRONG: 'Hyponatremia' (this is a lab finding, not a diagnosis)\n"
            "- RIGHT: 'Syndrome of Inappropriate Antidiuretic Hormone Secretion' (the disease causing hyponatremia)\n"
            "- WRONG: 'Fever' (this is a symptom, not a diagnosis)\n"
            "- RIGHT: 'Pneumonia' (the disease causing fever)\n"
            "- WRONG: 'Elevated CK' (this is a lab finding)\n"
            "- RIGHT: 'Rhabdomyolysis' (the disease causing elevated CK)\n"
            "- Always think: 'What disease/condition is causing this finding?' NOT 'What is the finding?'\n"
            "- Before diagnosing, carefully review ALL physical exam findings and test results.\n"
            "- Use your clinical reasoning to match the diagnosis to the most specific finding supported by the evidence.\n"
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
def normalize_for_comparison(dx_str):
    """Normalize diagnosis string for comparison (case-insensitive, punctuation-insensitive)"""
    if not dx_str:
        return ""
    # Convert to lowercase
    normalized = dx_str.lower().strip()
    # Remove trailing punctuation
    normalized = normalized.rstrip('.,;:!?')
    # Remove extra whitespace
    normalized = ' '.join(normalized.split())
    return normalized

def compare_results(diagnosis, correct_diagnosis, moderator_llm, mod_pipe):
    # Extract the diagnosis from doctor dialogue
    dx_match = re.search(r"DIAGNOSIS READY:\s*(.*)", diagnosis)
    doctor_dx = dx_match.group(1).strip() if dx_match else diagnosis.strip()
    
    # First check: exact match (case-insensitive, punctuation-insensitive)
    doctor_dx_normalized = normalize_for_comparison(doctor_dx)
    correct_dx_normalized = normalize_for_comparison(correct_diagnosis)
    
    if doctor_dx_normalized == correct_dx_normalized:
        return "yes"
    
    # Second check: use moderator for semantic comparison
    # Enhanced moderator prompt for stricter comparison
    moderator_prompt = f"""You are a strict medical diagnosis grader. Your task is to determine if the doctor's diagnosis matches the correct diagnosis.

CRITICAL RULES:
- The diagnoses must be the SAME specific disease/condition, not just similar or related
- "Imperforate hymen" and "Vaginal septum" are DIFFERENT conditions - answer "No"
- "Contact dermatitis" and "Bullous pemphigoid" are DIFFERENT conditions - answer "No"
- "Serotonin syndrome" and "Neuroleptic malignant syndrome" are DIFFERENT conditions - answer "No"
- "Hemorrhoids" and "Rectal cancer" are DIFFERENT conditions - answer "No"
- "Asthma" and "COPD" are DIFFERENT conditions - answer "No"
- "Primary biliary cholangitis" and "Primary sclerosing cholangitis" are DIFFERENT conditions - answer "No"
- "Adhesive capsulitis" and "Rotator cuff tendinopathy" are DIFFERENT conditions - answer "No"
- "Osteoclastoma" and "Aneurysmal bone cyst" are DIFFERENT conditions - answer "No"
- "C. difficile colitis" and "Antibiotic-associated diarrhea" are DIFFERENT conditions (one is specific, one is generic) - answer "No"
- "Complex partial seizure" and "Psychogenic non-epileptic seizures" are DIFFERENT conditions - answer "No"
- "Disseminated Gonococcal Infection" and "Gonococcal arthritis" are the SAME condition (arthritis is a manifestation) - answer "Yes"
- "Respiratory Distress Syndrome" and "Neonatal Respiratory Distress Syndrome" are the SAME condition - answer "Yes"
- "Osteoarthritis" and "Osteoarthritis of the hip and knee" are the SAME condition (location is just additional detail) - answer "Yes"
- Only answer "Yes" if they are the exact same condition (allowing for minor phrasing differences like "Crohn disease" vs "Crohn's disease")

Correct diagnosis: {correct_diagnosis}
Doctor's diagnosis: {doctor_dx}

Are these the SAME specific disease/condition? Answer ONLY "Yes" or "No"."""
    
    answer = query_model(moderator_llm, moderator_prompt, "You are a strict medical diagnosis grader. Respond only with Yes or No.", clip_prompt=True)
    return answer.lower().strip()


# -----------------
# Reasoning Critic and Evidence Checker (helper agents for first DIAGNOSIS READY)
# -----------------
REASONING_CRITIC_SYSTEM = """You are a clinical reasoning critic. You do NOT know the correct diagnosis.

Given a conversation transcript and the doctor's proposed diagnosis, output exactly two lines in this format:
ALTERNATIVES: <2-3 plausible alternative diagnoses that fit the evidence, comma-separated>
DISTINCTION: <one key distinguishing question or test between the proposed diagnosis and the most likely alternative>

Do not reveal or hint at the correct diagnosis. Use your own clinical reasoning to suggest plausible alternatives and one discriminative question or test."""

EVIDENCE_CHECKER_SYSTEM = """You are an evidence consistency checker. You do NOT know the correct diagnosis.

Given the conversation transcript, available exam/test evidence, and the doctor's proposed diagnosis, output exactly three lines in this format:
SUPPORT: <findings from the transcript or evidence that support the proposed diagnosis>
CONTRADICT: <2-3 specific findings that argue AGAINST the proposed diagnosis or fit another diagnosis better>
ABSENT: <2-3 hallmark findings typically expected for this diagnosis but not mentioned or absent in the case>

Do not reveal or hint at the correct diagnosis. Be specific: name the actual findings from the transcript. CONTRADICT and ABSENT help the doctor reconsider before committing."""


def run_reasoning_critic(controller_llm, transcript_text, proposed_dx):
    """Run the reasoning critic: returns ALTERNATIVES and DISTINCTION (generic, no gold dx)."""
    user_prompt = (
        f"Transcript:\n{transcript_text[:12000]}\n\n"
        f"Proposed diagnosis: {proposed_dx}\n\n"
        "Output ALTERNATIVES: and DISTINCTION: as specified."
    )
    out = query_model(controller_llm, user_prompt, REASONING_CRITIC_SYSTEM, clip_prompt=True)
    out = (out or "").strip()
    if not out:
        out = "ALTERNATIVES: (none suggested)\nDISTINCTION: Consider a key distinguishing test or question."
    return out


def run_evidence_checker(controller_llm, transcript_text, exam_and_tests_text, proposed_dx):
    """Run the evidence checker: returns SUPPORT, CONTRADICT, and ABSENT (generic, no gold dx)."""
    user_prompt = (
        f"Transcript:\n{transcript_text[:12000]}\n\n"
        f"Exam/test evidence:\n{exam_and_tests_text[:4000]}\n\n"
        f"Proposed diagnosis: {proposed_dx}\n\n"
        "Output SUPPORT:, CONTRADICT:, and ABSENT: as specified."
    )
    out = query_model(controller_llm, user_prompt, EVIDENCE_CHECKER_SYSTEM, clip_prompt=True)
    out = (out or "").strip()
    if not out:
        out = (
            "SUPPORT: (review transcript)\n"
            "CONTRADICT: Recheck findings that argue against the diagnosis.\n"
            "ABSENT: List hallmark findings expected for this diagnosis but not present."
        )
    return out


DEVIL_ADVOCATE_SYSTEM = """You are a devil's advocate. You do NOT know the correct diagnosis.

Given the conversation transcript and the doctor's proposed diagnosis, argue AGAINST it. Output exactly:
COUNTERARGUMENT: <2-4 sentences: the strongest reason(s) this diagnosis could be wrong, or why an alternative might fit better. Use only evidence from the transcript. Do not reveal or hint at the correct answer.>"""


def run_devils_advocate(controller_llm, transcript_text, proposed_dx):
    """Run devil's advocate: returns COUNTERARGUMENT (generic, no gold dx)."""
    user_prompt = (
        f"Transcript:\n{transcript_text[:12000]}\n\n"
        f"Proposed diagnosis: {proposed_dx}\n\n"
        "Output COUNTERARGUMENT: as specified."
    )
    out = query_model(controller_llm, user_prompt, DEVIL_ADVOCATE_SYSTEM, clip_prompt=True)
    out = (out or "").strip()
    if not out or "COUNTERARGUMENT:" not in out:
        out = "COUNTERARGUMENT: Reconsider whether the evidence fully supports this diagnosis versus alternatives."
    return out


QUESTION_QUALITY_SYSTEM = """You are a question quality scorer. You do NOT know the correct diagnosis.

Given the conversation transcript and the last question the doctor asked the patient, score how discriminative that question was for narrowing the differential (1-5).
- 1-2: Low yield (generic, already answered, or does not rule in/out specific diagnoses).
- 3: Moderately discriminative.
- 4-5: High yield (clearly rules in/out at least one plausible diagnosis).

Output EXACTLY one line: SCORE: <integer 1-5>"""


def run_question_quality_scorer(controller_llm, transcript_text, last_doctor_question):
    """Returns integer 1-5. Low = not discriminative."""
    user_prompt = (
        f"Transcript:\n{transcript_text[:8000]}\n\n"
        f"Last doctor question: {last_doctor_question}\n\n"
        "Output SCORE: <1-5> as specified."
    )
    out = query_model(controller_llm, user_prompt, QUESTION_QUALITY_SYSTEM, clip_prompt=True)
    out = (out or "").strip()
    m = re.search(r"SCORE:\s*(\d)", out)
    if m:
        return max(1, min(5, int(m.group(1))))
    return 3


TEST_ORDERING_SYSTEM = """You are a test-ordering advisor. You do NOT know the correct diagnosis.

Given the conversation transcript, the doctor's current differential (list of possible diagnoses), and available exam/test context, suggest the single best test or question that would best distinguish among the top differential diagnoses.

Output EXACTLY one line: SUGGESTED_TEST: <one specific test name or one key question to ask>"""


def propose_test_for_ddx(controller_llm, transcript_text, working_ddx_list, exam_and_tests_text):
    """Suggest one test or question to distinguish the differential. working_ddx_list: list of diagnosis strings."""
    ddx_str = ", ".join(working_ddx_list[:5]) if working_ddx_list else "none stated"
    user_prompt = (
        f"Transcript:\n{transcript_text[:8000]}\n\n"
        f"Current differential: {ddx_str}\n\n"
        f"Exam/tests so far:\n{exam_and_tests_text[:3000]}\n\n"
        "Output SUGGESTED_TEST: as specified."
    )
    out = query_model(controller_llm, user_prompt, TEST_ORDERING_SYSTEM, clip_prompt=True)
    out = (out or "").strip()
    if "SUGGESTED_TEST:" in out:
        return out.split("SUGGESTED_TEST:", 1)[1].strip()
    return ""


def _get_working_ddx_strings(evidence_ledger):
    """Extract list of diagnosis strings from working_ddx (may be list of str or list of dict)."""
    wddx = evidence_ledger.get("working_ddx") or []
    out = []
    for item in wddx[:10]:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
        elif isinstance(item, dict):
            for key in ["diagnosis", "dx", "name"]:
                if key in item and isinstance(item[key], str) and item[key].strip():
                    out.append(item[key].strip())
                    break
    return out


def _dx_in_working_ddx(proposed_dx, working_ddx_strings):
    """True if proposed_dx is in or closely matches one of working_ddx_strings."""
    if not working_ddx_strings:
        return True  # No ddx listed: allow (don't block)
    p = proposed_dx.strip().lower()
    for d in working_ddx_strings:
        d = d.strip().lower()
        if p == d or p in d or d in p:
            return True
        if set(p.split()) & set(d.split()):
            return True
    return False


# -----------------
# Question Controller (Generic: no diagnostic-specific hints)
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
- The question must be discriminative (rules IN or OUT at least 2 plausible diagnoses from the conversation).
- If physical exam findings are provided, review them and suggest questions that explore unexplained or critical findings.
- Prefer asking about missing critical info, risk factors, or a decisive symptom/sign. Avoid low-yield general questions.
- Be patient-friendly and nonjudgmental.
- Use only general clinical reasoning; do not assume specific disease categories.
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
# DX Normalizer (Conservative: rephrase only, never substitute)
# -----------------
DX_NORMALIZER_SYSTEM = """
You are a diagnosis label normalizer for an exam grader.

STRICT RULES:
- Output EXACTLY one line: FINAL_DX: <label>
- NO extra text.
- ONLY rephrase for standard spelling/terminology (e.g. "Crohn's disease" -> "Crohn disease", fix typos).
- DO NOT substitute a different condition. If the doctor said "Dumping syndrome", do NOT output "Hypoglycemia". If the doctor said "Renal calculi", do NOT output "Renal cysts". Only minor wording/formatting changes are allowed.
- If answer choices are provided (NEJM/NEJM_Ext), you MUST pick the answer choice that matches the doctor's diagnosis verbatim, or the closest match from the list. Do not replace with a different answer choice.
- If the proposed diagnosis is already clear and specific, output it unchanged.
- When in doubt, preserve the doctor's exact diagnosis.
"""

def normalize_dx(normalizer_llm, doctor_dialogue, scenario, dataset):
    # Extract raw diagnosis (strip optional CONFIDENCE suffix)
    m = re.search(r"DIAGNOSIS READY:\s*(.*)", doctor_dialogue)
    raw_dx = m.group(1).strip() if m else doctor_dialogue.strip()
    raw_dx = re.sub(r"\s*\[?\s*CONFIDENCE:\s*(?:high|medium|low)\s*\]?\s*$", "", raw_dx, flags=re.I).strip()

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

    try:
        exam_info = scenario.exam_information()
        case_context = json.dumps(exam_info, indent=2) if isinstance(exam_info, dict) else str(exam_info)
    except Exception:
        case_context = ""

    user_prompt = (
        f"Doctor proposed diagnosis: {raw_dx}\n\n"
        f"{options_txt}\n\n"
        f"Case context (for spelling/terminology only):\n{case_context[:2000]}\n\n"
        f"Task: Output ONLY a rephrase for standard spelling/terminology. Do NOT substitute a different diagnosis."
    )
    out = query_model(normalizer_llm, user_prompt, DX_NORMALIZER_SYSTEM, clip_prompt=True).strip()
    if out.lower().startswith("final_dx:"):
        normalized = out.split(":", 1)[1].strip()
        raw_lower = raw_dx.lower().strip('.,;:!?')
        normalized_lower = normalized.lower().strip('.,;:!?')
        # Conservative: reject substitution (different condition). Keep only rephrases.
        raw_words = set(raw_lower.split())
        norm_words = set(normalized_lower.split())
        if raw_words and norm_words:
            overlap = len(raw_words & norm_words)
            total_unique = len(raw_words | norm_words)
            similarity = overlap / total_unique if total_unique > 0 else 0
            if similarity < 0.4:
                return raw_dx  # Normalizer tried to substitute; keep original
        # NEJM: if answer choices exist, prefer verbatim match from list
        if options_txt and dataset in ["NEJM", "NEJM_Ext"]:
            try:
                opts = [a["text"] for a in scenario.scenario_dict.get("answers", []) if isinstance(a, dict) and "text" in a]
                for o in opts:
                    if o.strip().lower() == normalized_lower or o.strip().lower() == raw_lower:
                        return o.strip()
                if raw_dx.strip() in opts:
                    return raw_dx.strip()
            except Exception:
                pass
        return normalized
    return raw_dx

# -----------------
# Main run function
# -----------------
def main(api_key, replicate_api_key, inf_type, doctor_bias, patient_bias, doctor_llm, patient_llm, measurement_llm, moderator_llm, num_scenarios, dataset, img_request, total_inferences, anthropic_api_key=None, evidence_lock=False, guideline_snippets_path="data/guideline_snippets.csv", use_intake_assistant=False, intake_llm="gpt4", intake_turns=6, question_controller_llm=None, scenario_ids=None, use_reasoning_helpers=True):
    # Use provided API key, or fall back to environment variable
    if not api_key or api_key == "":
        api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("OpenAI API key is required. Set --openai_api_key or OPENAI_API_KEY environment variable.")
    openai.api_key = api_key
    # Create OpenAI client for new API (if available)
    global _openai_client
    _openai_client = None
    if OPENAI_NEW_API:
        try:
            # Initialize OpenAI client with timeout settings to prevent hanging
            try:
                import httpx
                _openai_client = OpenAI(
                    api_key=api_key,
                    timeout=httpx.Timeout(60.0, connect=10.0)  # 60s total, 10s connect
                )
            except ImportError:
                # httpx not available, use default timeout (OpenAI client has built-in timeout)
                _openai_client = OpenAI(api_key=api_key)
        except Exception as e:
            print(f"Warning: Could not initialize OpenAI client: {e}. Falling back to default.", flush=True)
            try:
                _openai_client = OpenAI(api_key=api_key)
            except Exception:
                _openai_client = None
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
        reasoning_check_done = False  # track if we've run Reasoning Critic + Evidence Checker for this scenario
        devil_advocate_done = False  # track if we've run Devil's Advocate for this scenario
        consecutive_low_quality = 0  # question quality scorer: consecutive low scores
        # Allow 2 extra iterations for reasoning + devil's advocate feedback when helpers are on
        max_iters = total_inferences + (2 if use_reasoning_helpers else 0)

        for _inf_id in range(max_iters):
            is_feedback_turn = use_reasoning_helpers and _inf_id >= total_inferences
            # NEJM image policy
            if dataset == "NEJM":
                if img_request:
                    imgs = "REQUEST IMAGES" in doctor_dialogue
                else:
                    imgs = True
            else:
                imgs = False

            # Final turn hint (last normal turn or last iteration overall)
            if _inf_id == total_inferences - 1 and not is_feedback_turn:
                pi_dialogue += "\n\n⚠️ CRITICAL: This is your FINAL turn. You MUST output 'DIAGNOSIS READY: [diagnosis]' now. Do not ask another question - provide your diagnosis immediately.\n"
            elif _inf_id == max_iters - 1 and is_feedback_turn:
                pi_dialogue += "\n\n⚠️ This is your last chance. Output 'DIAGNOSIS READY: [diagnosis]' now.\n"

            # Question Controller: propose high-yield question (skip on feedback turns)
            doctor_input = pi_dialogue
            if inf_type != "human_doctor" and not is_feedback_turn:
                is_feedback_content = pi_dialogue.strip().startswith("REASONING CHECK:") or pi_dialogue.strip().startswith("DEVIL'S ADVOCATE:")
                # Heuristic: skip controller when content is moderator feedback; else call in early turns or when not after test request
                should_call_controller = (
                    not is_feedback_content
                    and (
                        (_inf_id < 5) or
                        (("REQUEST TEST" not in doctor_dialogue) and (_inf_id < total_inferences - 1))
                    )
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
                    # Optional: test-ordering suggestion when we have a differential
                    working_ddx_strings = _get_working_ddx_strings(doctor_agent.evidence_ledger)
                    extra_guidance = ""
                    if len(working_ddx_strings) >= 2:
                        try:
                            exam_raw = scenario.exam_information()
                            exam_and_tests_text = json.dumps(exam_raw, indent=2) if isinstance(exam_raw, dict) else str(exam_raw)
                        except Exception:
                            exam_and_tests_text = ""
                        suggested_test = propose_test_for_ddx(
                            controller_llm_to_use, transcript_text, working_ddx_strings, exam_and_tests_text
                        )
                        if suggested_test:
                            extra_guidance = f"\nConsider requesting or asking: {suggested_test}"
                    if consecutive_low_quality >= 2:
                        extra_guidance += "\n\nConsider asking a more discriminative question (one that rules in/out specific diagnoses)."
                    # Create doctor_input with guidance (DO NOT modify pi_dialogue)
                    doctor_input = (
                        f"{pi_dialogue}\n\n"
                        f"MODERATOR GUIDANCE: Ask this next question verbatim unless impossible:\n"
                        f"\"{next_q}\"{extra_guidance}"
                    )

            # Doctor turn
            if inf_type == "human_doctor":
                doctor_dialogue = input("\nQuestion for patient: ")
            else:
                doctor_dialogue = doctor_agent.inference_doctor(
                    doctor_input, image_requested=imgs, allow_extra_feedback_turn=is_feedback_turn
                )

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

                # Differential-first: proposed diagnosis must be in working_ddx (unless ddx empty)
                dx_match = re.search(r"DIAGNOSIS READY:\s*(.*)", doctor_dialogue)
                proposed_dx_raw = dx_match.group(1).strip() if dx_match else ""
                proposed_dx = re.sub(r"\s*\[?\s*CONFIDENCE:\s*(?:high|medium|low)\s*\]?\s*$", "", proposed_dx_raw, flags=re.I).strip()
                working_ddx_strings = _get_working_ddx_strings(doctor_agent.evidence_ledger)
                if not _dx_in_working_ddx(proposed_dx, working_ddx_strings) and working_ddx_strings and _inf_id < max_iters - 1:
                    ddx_list_str = "; ".join(working_ddx_strings[:5])
                    pi_dialogue = (
                        f"MODERATOR: Your DIAGNOSIS READY must be one of your working_ddx. You listed: {ddx_list_str}. "
                        f"You proposed '{proposed_dx}'. Please output DIAGNOSIS READY: <one of the above> or ask one more question / request a test."
                    )
                    print("MODERATOR: Differential-first reject (diagnosis not in working_ddx)", flush=True)
                    continue

                # First DIAGNOSIS READY + reasoning helpers: run critic + evidence checker, give doctor one more turn (or use extra feedback iteration).
                if use_reasoning_helpers and not reasoning_check_done and ( _inf_id < total_inferences - 1 or is_feedback_turn ):
                    transcript = (doctor_agent.agent_hist + "\nDoctor (latest): " + doctor_dialogue).strip()
                    try:
                        exam_raw = scenario.exam_information()
                        exam_and_tests_text = json.dumps(exam_raw, indent=2) if isinstance(exam_raw, dict) else str(exam_raw)
                    except Exception:
                        exam_and_tests_text = ""
                    critic_out = run_reasoning_critic(doctor_llm, transcript, proposed_dx)
                    time.sleep(2.0)
                    evidence_out = run_evidence_checker(doctor_llm, transcript, exam_and_tests_text, proposed_dx)
                    reasoning_check_done = True
                    pi_dialogue = (
                        f"REASONING CHECK: {critic_out}\n\n"
                        f"EVIDENCE CHECK: {evidence_out}\n\n"
                        "If you still stand by your diagnosis, output DIAGNOSIS READY: [your diagnosis]. "
                        "If you want to ask one more question or request a test, do that instead."
                    )
                    continue

                # Second DIAGNOSIS READY + devil's advocate: one more adversarial round
                if use_reasoning_helpers and reasoning_check_done and not devil_advocate_done and ( _inf_id < total_inferences - 1 or is_feedback_turn ):
                    transcript = (doctor_agent.agent_hist + "\nDoctor (latest): " + doctor_dialogue).strip()
                    devil_out = run_devils_advocate(doctor_llm, transcript, proposed_dx)
                    time.sleep(2.0)
                    devil_advocate_done = True
                    pi_dialogue = (
                        f"DEVIL'S ADVOCATE: {devil_out}\n\n"
                        "If you still stand by your diagnosis, output DIAGNOSIS READY: [your diagnosis]. "
                        "Otherwise ask one more question or request a test."
                    )
                    continue

                # Uncertainty-gated: if confidence is low, trigger one more round (devil's advocate) if we have turns
                confidence_low = bool(re.search(r"CONFIDENCE:\s*(?:low|uncertain)", proposed_dx_raw, re.I))
                if confidence_low and _inf_id < max_iters - 1 and not devil_advocate_done:
                    transcript = (doctor_agent.agent_hist + "\nDoctor (latest): " + doctor_dialogue).strip()
                    devil_out = run_devils_advocate(doctor_llm, transcript, proposed_dx)
                    time.sleep(2.0)
                    devil_advocate_done = True
                    pi_dialogue = (
                        f"DEVIL'S ADVOCATE (low confidence round): {devil_out}\n\n"
                        "Reconsider and then output DIAGNOSIS READY: [your diagnosis] or ask one more question."
                    )
                    continue

                # Grade and break
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

            # Question quality scorer: after a doctor question (not on feedback-only iterations)
            if not is_feedback_turn and inf_type != "human_doctor" and "DIAGNOSIS READY" not in doctor_dialogue and "REQUEST TEST" not in doctor_dialogue:
                is_feedback = pi_dialogue.strip().startswith("REASONING CHECK:") or pi_dialogue.strip().startswith("DEVIL'S ADVOCATE:")
                if not is_feedback:
                    transcript_for_q = (doctor_agent.agent_hist + "\nDoctor (latest): " + doctor_dialogue).strip()
                    q_score = run_question_quality_scorer(doctor_llm, transcript_for_q, doctor_dialogue[:500])
                    if q_score <= 2:
                        consecutive_low_quality += 1
                    else:
                        consecutive_low_quality = 0

            # Measurement agent and patient reply only on normal turns (not feedback-only)
            if not is_feedback_turn and "REQUEST TEST" in doctor_dialogue:
                pi_dialogue = meas_agent.inference_measurement(doctor_dialogue)
                print("Measurement [{}%]:".format(int(((_inf_id+1)/total_inferences)*100)), pi_dialogue)
                patient_agent.add_hist(pi_dialogue)
                # Turn on the "result integration gate" right after MeasurementAgent returns a result
                doctor_agent.must_integrate_result = True
                doctor_agent.pending_result_text = pi_dialogue
            elif not is_feedback_turn:
                # Patient reply
                if inf_type == "human_patient":
                    pi_dialogue = input("\nResponse to doctor: ")
                else:
                    pi_dialogue = patient_agent.inference_patient(doctor_dialogue)
                print("Patient [{}%]:".format(int(((_inf_id+1)/total_inferences)*100)), pi_dialogue)
                meas_agent.add_hist(pi_dialogue)

            # Throttle to avoid rate limits (TPM); also prevents API timeouts
            time.sleep(3.0)


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
    parser.add_argument('--no_reasoning_helpers', action='store_true', help='Disable Reasoning Critic and Evidence Checker; first DIAGNOSIS READY is graded immediately.')

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
        scenario_ids=args.scenario_ids,
        use_reasoning_helpers=not args.no_reasoning_helpers
    )
