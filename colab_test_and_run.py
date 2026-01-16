# ============================================================================
# Google Colab - Test API Connection & Run AgentClinic Enhanced
# ============================================================================
# This version tests the API first, then runs with better diagnostics
# ============================================================================

# Step 1: Install dependencies
print("📦 Installing dependencies...")
!pip install -q openai==0.28.0 anthropic replicate transformers datasets regex jsonschema
print("✓ Dependencies installed\n")

# Step 2: Set API key
import os
OPENAI_API_KEY = "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"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
print("✓ API key set\n")

# Step 3: Test API connection first
print("🔍 Testing OpenAI API connection...")
import openai
openai.api_key = OPENAI_API_KEY

try:
    test_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Say 'API test successful'"}],
        max_tokens=10,
        timeout=10
    )
    print(f"✓ API test successful: {test_response['choices'][0]['message']['content']}\n")
except Exception as e:
    print(f"✗ API test failed: {e}\n")
    print("Please check your API key and try again.")
    raise

# Step 4: Upload required files
print("📁 Please upload the following files:")
print("   - agentclinic_enhanced.py")
print("   - evidence_block.py")
print("   - rag.py")
print("   - agentclinic_medqa_extended.jsonl")
print("   - data/guideline_snippets.csv (optional)\n")

from google.colab import files
print("Upload files now:")
uploaded = files.upload()

# Fix file names (Colab sometimes adds numbers)
import shutil
if 'agentclinic_enhanced (1).py' in uploaded:
    shutil.move('agentclinic_enhanced (1).py', 'agentclinic_enhanced.py')
if 'rag (1).py' in uploaded:
    shutil.move('rag (1).py', 'rag.py')
if 'guideline_snippets (1).csv' in uploaded:
    shutil.move('guideline_snippets (1).csv', 'guideline_snippets.csv')

# Create data directory if needed
!mkdir -p data
if os.path.exists('guideline_snippets.csv'):
    !mv guideline_snippets.csv data/ 2>/dev/null || true

print("\n✓ Files uploaded\n")

# Step 5: Verify files
import os
required_files = [
    'agentclinic_enhanced.py',
    'evidence_block.py',
    'rag.py',
    'agentclinic_medqa_extended.jsonl'
]

missing = []
for f in required_files:
    if os.path.exists(f):
        print(f"  ✓ {f}")
    else:
        print(f"  ✗ {f} MISSING")
        missing.append(f)

if missing:
    print(f"\n⚠ Missing files: {missing}")
    print("Please upload them and run this cell again.")
else:
    print("\n✓ All required files present\n")
    
    # Step 6: Run with 1 scenario first (faster test)
    print("🚀 Running AgentClinic Enhanced on MedQA Extended...")
    print("Starting with 1 scenario for testing (increase --num_scenarios after it works)\n")
    print("=" * 70)
    
    !python agentclinic_enhanced.py \
      --openai_api_key $OPENAI_API_KEY \
      --agent_dataset MedQA_Ext \
      --doctor_llm gpt4 \
      --patient_llm gpt4 \
      --measurement_llm gpt4 \
      --moderator_llm gpt4 \
      --use_intake_assistant \
      --intake_assistant_llm gpt4 \
      --intake_assistant_turns 3 \
      --num_scenarios 1 \
      --total_inferences 10
    
    print("\n" + "=" * 70)
    print("✅ Done! If this worked, increase --num_scenarios to 5 or more.")

