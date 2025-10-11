# Day 2 Checklist - CSM Setup

## 🎯 Today's Goal
Get CSM running and measure baseline latency

---

## ✅ Step-by-Step Checklist

### Part 1: Hugging Face Setup (15 min)

- [ ] **1.1** Go to https://huggingface.co/join
- [ ] **1.2** Create account and verify email
- [ ] **1.3** Go to https://huggingface.co/sesame-ai/csm-1b
- [ ] **1.4** Click "Request Access" (usually instant approval)
- [ ] **1.5** Go to https://huggingface.co/settings/tokens
- [ ] **1.6** Create new token (name: "oviya-csm", type: Read)
- [ ] **1.7** Copy token (starts with `hf_...`)
- [ ] **1.8** Add to .env file:
  ```bash
  echo "HUGGINGFACE_TOKEN=hf_your_token_here" >> .env
  ```

### Part 2: Choose Your Setup (Pick ONE)

#### Option A: RunPod (Recommended) ⭐

- [ ] **2A.1** Go to https://www.runpod.io/
- [ ] **2A.2** Sign up and add payment method
- [ ] **2A.3** Deploy Pod:
  - Template: PyTorch 2.0
  - GPU: RTX A4000 (16GB, $0.34/hr)
  - Disk: 50GB
  - Expose port: 8000
- [ ] **2A.4** Connect via SSH or Jupyter Lab
- [ ] **2A.5** In Pod terminal, run:
  ```bash
  cd /workspace
  git clone https://github.com/SesameAILabs/csm.git
  cd csm
  pip install -r requirements.txt
  huggingface-cli login  # Paste your token
  python run_csm.py
  ```

#### Option B: Local GPU

- [ ] **2B.1** Check GPU: `nvidia-smi`
- [ ] **2B.2** Clone CSM:
  ```bash
  cd validation/csm-benchmark
  git clone https://github.com/SesameAILabs/csm.git
  cd csm
  ```
- [ ] **2B.3** Install dependencies:
  ```bash
  pip install -r requirements.txt
  huggingface-cli login
  ```
- [ ] **2B.4** Test:
  ```bash
  python run_csm.py
  ```

### Part 3: Run Benchmark Tests (30 min)

- [ ] **3.1** Navigate to test directory:
  ```bash
  cd /Users/jarvis/Documents/Oviya\ EI/oviya-ai/validation/csm-benchmark
  ```

- [ ] **3.2** Run basic test:
  ```bash
  python basic_test.py
  ```

- [ ] **3.3** Check results:
  - Cold start < 30s? ✅/❌
  - Warm latency < 2s? ✅/❌
  - Audio files generated? ✅/❌

- [ ] **3.4** Listen to audio files:
  ```bash
  open results/quality_test.wav
  ```

- [ ] **3.5** Rate audio quality (1-5): _____

### Part 4: Document Results (10 min)

- [ ] **4.1** Check `results/day_2_results.txt`
- [ ] **4.2** Note any issues or observations
- [ ] **4.3** Update PROGRESS.md with Day 2 status

---

## 🎯 Success Criteria

All must pass to proceed to Day 3:

- ✅ CSM model loads successfully
- ✅ Cold start < 30 seconds
- ✅ Warm inference < 2 seconds
- ✅ Audio quality acceptable (3+/5)

---

## 💰 Cost Tracking

**Budget for Day 2**: $2.00

| Item | Estimated Cost |
|------|----------------|
| RunPod (2 hours) | $0.68-1.52 |
| Hugging Face | Free |
| **Total** | **$0.68-1.52** |

---

## 🐛 Common Issues

### Issue: "Access denied to model"
**Fix**: Make sure you requested access at https://huggingface.co/sesame-ai/csm-1b

### Issue: "CUDA out of memory"
**Fix**: Use RunPod with A40 (48GB) instead of A4000

### Issue: "No module named 'generator'"
**Fix**: Make sure you're in the csm directory

### Issue: Slow generation (>5s)
**Fix**: Make sure you're using GPU, not CPU

---

## 📞 Need Help?

If you get stuck:
1. Check the error message carefully
2. Look in `validation/csm-benchmark/SETUP.md`
3. Try the troubleshooting section
4. Ask for help with the specific error

---

## 🔜 After Day 2

Once all tests pass, you're ready for:

**Day 3**: Audio Context Emotion Testing
- Record emotion samples
- Test audio context approach
- Measure latency overhead

---

## ⏱️ Estimated Time

- Hugging Face setup: 15 min
- RunPod/Local setup: 30 min
- CSM installation: 15 min
- Running tests: 30 min
- Documentation: 10 min

**Total: ~1.5-2 hours**

---

**Good luck! 🚀**

