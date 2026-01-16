# ============================================================================
# Google Colab - Run AgentClinic Enhanced on MedQA Extended
# ============================================================================
# Paste this entire cell into Google Colab and run it
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

# Step 3: Upload required files
print("📁 Please upload the following files:")
print("   - agentclinic_enhanced.py")
print("   - evidence_block.py")
print("   - rag.py")
print("   - agentclinic_medqa_extended.jsonl")
print("   - data/guideline_snippets.csv (optional, for evidence_lock)\n")

from google.colab import files
print("Upload files now (you'll get file picker dialogs):")
uploaded = files.upload()

# Create data directory if needed
!mkdir -p data
if 'guideline_snippets.csv' in uploaded:
    !mv guideline_snippets.csv data/

print("\n✓ Files uploaded\n")

# Step 4: Verify files
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
    
    # Step 5: Run the enhanced script
    print("🚀 Running AgentClinic Enhanced on MedQA Extended (5 scenarios)...\n")
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
      --intake_assistant_turns 5 \
      --num_scenarios 5 \
      --total_inferences 10
    
    print("\n" + "=" * 70)
    print("✅ Done!")

