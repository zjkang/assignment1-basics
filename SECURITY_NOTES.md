# 🔒 Security Notes

## ⚠️ IMPORTANT: API Key Management

### **Never Commit API Keys to Git!**

Your wandb API key has been **removed** from all documentation files and replaced with placeholders.

---

## ✅ **Recommended Setup**

### **Option 1: Add to Shell Config (Recommended)**

Add this line to your `~/.zshrc` or `~/.bashrc`:

```bash
export WANDB_API_KEY="your_actual_api_key_here"
```

Then reload:
```bash
source ~/.zshrc
```

**Benefits**:
- ✅ Available in all terminal sessions
- ✅ Never committed to git
- ✅ Easy to use

---

### **Option 2: Use .env File (Alternative)**

1. Create a `.env` file in the project root:
   ```bash
   echo 'WANDB_API_KEY="your_actual_api_key_here"' > .env
   ```

2. The `.env` file is already in `.gitignore` (as `.env.local`)

3. Load it before running:
   ```bash
   source .env
   uv run python scripts/train_with_wandb.py
   ```

---

### **Option 3: Set Temporarily (For Testing)**

```bash
export WANDB_API_KEY="your_actual_api_key_here"
uv run python scripts/train_with_wandb.py
```

**Note**: This only lasts for the current terminal session.

---

## 🔍 **What's in .gitignore**

The following are now ignored and won't be committed:

### **Sensitive Data**
- `wandb/` - Run data and logs
- `.env*` - Environment variable files

### **Large Files**
- `checkpoints/` - Model checkpoints
- `data/` - Training data
- `tokenizer/*.pkl` - Tokenizer files (optional)

### **Cache/Temp Files**
- `__pycache__/`
- `.pytest_cache/`
- `*.pyc`
- `*.log`

### **Editor Files**
- `.vscode/`
- `.idea/`
- `.DS_Store`

---

## ✅ **Before You Commit**

Run this checklist:

```bash
# 1. Check for sensitive data
git diff

# 2. Verify no API keys
grep -r "wandb_v1_" . --exclude-dir=.git

# 3. Check .gitignore is working
git status

# 4. Verify large files aren't staged
du -sh .git/

# 5. Safe to commit!
git add .
git commit -m "Your message"
```

---

## 🚨 **If You Already Committed API Key**

If you accidentally committed your API key:

### **Step 1: Revoke the Old Key**
1. Go to https://wandb.ai/settings
2. Under "API keys", click "Revoke" on the exposed key
3. Generate a new key

### **Step 2: Remove from Git History**
```bash
# WARNING: This rewrites history!
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch FILENAME" \
  --prune-empty --tag-name-filter cat -- --all
```

**Or use BFG Repo-Cleaner** (easier):
```bash
brew install bfg  # macOS
bfg --replace-text passwords.txt  # Create passwords.txt with your API key
```

### **Step 3: Force Push** (⚠️ Dangerous!)
```bash
git push --force --all
```

**⚠️ Only do this if no one else has cloned your repo!**

---

## 📝 **Summary**

✅ **DO**:
- Use environment variables for API keys
- Add sensitive files to `.gitignore`
- Review `git diff` before committing

❌ **DON'T**:
- Hardcode API keys in code or docs
- Commit large files (checkpoints, data)
- Commit wandb run directories

---

## 🔗 **More Resources**

- [GitHub: Removing Sensitive Data](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository)
- [Wandb: Security Best Practices](https://docs.wandb.ai/guides/technical-faq/general#how-secure-is-wandb)

---

**Stay Safe!** 🔒
