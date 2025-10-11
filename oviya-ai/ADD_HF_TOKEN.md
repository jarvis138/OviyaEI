# Add Your Hugging Face Token

## Quick Steps:

### 1. Get Your Token

If you already have a Hugging Face account:
- Go to: https://huggingface.co/settings/tokens
- Click "New token"
- Name: "oviya-csm"
- Type: **Read**
- Click "Generate"
- Copy the token (starts with `hf_...`)

If you DON'T have an account yet:
- Go to: https://huggingface.co/join
- Sign up and verify email
- Then follow steps above

### 2. Request CSM Model Access

- Go to: https://huggingface.co/sesame-ai/csm-1b
- Click "Request Access"
- Usually approved instantly!

### 3. Add Token to .env

**Option A: Edit .env file directly**
```bash
# Open .env in your editor
open .env

# Replace this line:
HUGGINGFACE_TOKEN=your_token_here

# With your actual token:
HUGGINGFACE_TOKEN=hf_your_actual_token_here
```

**Option B: Use command line**
```bash
# Replace YOUR_TOKEN with your actual token
echo "HUGGINGFACE_TOKEN=hf_YOUR_TOKEN" >> .env
```

### 4. Test Access

```bash
cd validation
python test_hf_access.py
```

Expected output:
```
✅ Token found: hf_abc...xyz
✅ huggingface_hub installed
✅ Access granted to sesame-ai/csm-1b
🎉 All tests passed! Ready to install CSM!
```

---

## ⚠️ Important Notes

- **Never commit .env to git** (it's already in .gitignore)
- Keep your token secret
- Token should start with `hf_`
- If access denied, wait a few minutes for approval

---

## Next Steps

Once token is added and tested:

1. **Choose setup**: RunPod or Local GPU
2. **Follow**: `validation/csm-benchmark/SETUP.md`
3. **Run tests**: `python basic_test.py`

---

**Need help?** Check `validation/DAY-2-CHECKLIST.md`

