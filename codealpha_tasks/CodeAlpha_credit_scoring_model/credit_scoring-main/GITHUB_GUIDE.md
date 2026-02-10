# GitHub Deployment Guide - Credit Scoring Project

This guide will walk you through uploading your credit scoring project to GitHub step by step.

---

## 📋 Prerequisites

Before you begin, ensure you have:
- ✅ Git installed on your computer
- ✅ A GitHub account
- ✅ Your credit scoring project files ready

---

## 🔧 Step 1: Install Git (if not already installed)

### Windows
1. Download Git from https://git-scm.com/download/win
2. Run the installer with default settings
3. Open **Git Bash** from the Start menu

### macOS
```bash
# Using Homebrew
brew install git

# Or download from
# https://git-scm.com/download/mac
```

### Linux (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install git
```

### Verify Installation
```bash
git --version
# Should show: git version 2.x.x
```

---

## 👤 Step 2: Configure Git

Open terminal/Git Bash and run:

```bash
# Set your name
git config --global user.name "Your Name"

# Set your email (use your GitHub email)
git config --global user.email "your.email@example.com"

# Verify configuration
git config --list
```

---

## 🌐 Step 3: Create GitHub Repository

1. Go to https://github.com and log in
2. Click the **"+"** icon (top-right) → **"New repository"**
3. Fill in repository details:
   - **Repository name**: `credit_scoring`
   - **Description**: `Machine learning project for credit risk assessment and scoring`
   - **Visibility**: Choose **Public** or **Private**
   - ⚠️ **Important**: 
     - **DO NOT** check "Add a README file"
     - **DO NOT** add .gitignore
     - **DO NOT** choose a license yet
4. Click **"Create repository"**
5. **Keep this page open** - you'll need the repository URL

---

## 📂 Step 4: Prepare Your Project Directory

Navigate to your project folder:

```bash
# Example - adjust the path to match your project location
cd /path/to/credit_scoring

# Verify you're in the right directory
ls
# You should see: dataset.py, train.py, evaluate.py, etc.
```

---

## 🎯 Step 5: Initialize Git Repository

```bash
# Initialize git in your project
git init

# You should see:
# Initialized empty Git repository in /path/to/credit_scoring/.git/
```

---

## 📝 Step 6: Add Project Files to Git

```bash
# Check what files are in your directory
git status

# Add all files to staging area
git add .

# Verify files are staged
git status
# You should see files in green, ready to be committed
```

### What files will be added?

Your project includes:
- ✅ `dataset.py` - Dataset generation script
- ✅ `features.py` - Feature engineering script
- ✅ `train.py` - Model training script
- ✅ `evaluate.py` - Model evaluation script
- ✅ `README.md` - Project documentation
- ✅ `requirements.txt` - Python dependencies
- ✅ `.gitignore` - Files to ignore
- ✅ `credit_data.csv` - Dataset
- ✅ `credit_data_engineered.csv` - Engineered dataset
- ✅ `confusion_matrix.png` - Model visualization
- ✅ `saved/` directory - Trained models

---

## 💾 Step 7: Make Your First Commit

```bash
# Commit with a descriptive message
git commit -m "Initial commit: Add credit scoring ML project"

# You should see output showing files committed
```

---

## 🔗 Step 8: Connect to GitHub Repository

Copy your repository URL from GitHub (it looks like):
- HTTPS: `https://github.com/yourusername/credit_scoring.git`
- SSH: `git@github.com:yourusername/credit_scoring.git`

```bash
# Add GitHub as remote repository (replace with YOUR URL)
git remote add origin https://github.com/yourusername/credit_scoring.git

# Verify remote was added
git remote -v
# Should show:
# origin  https://github.com/yourusername/credit_scoring.git (fetch)
# origin  https://github.com/yourusername/credit_scoring.git (push)
```

---

## 🚀 Step 9: Push to GitHub

```bash
# Rename branch to 'main' (GitHub default)
git branch -M main

# Push your code to GitHub
git push -u origin main
```

### 🔐 Authentication Options

You'll be prompted to authenticate. Choose one:

#### Option A: Personal Access Token (Recommended)

1. **Generate a token**:
   - Go to GitHub → Settings → Developer settings
   - Click **Personal access tokens** → **Tokens (classic)**
   - Click **Generate new token (classic)**
   - Give it a name: "credit_scoring_upload"
   - Select scope: ✅ **repo** (full control of private repositories)
   - Click **Generate token**
   - **⚠️ COPY THE TOKEN IMMEDIATELY** (you won't see it again!)

2. **Use the token**:
   - When prompted for password, paste your token
   - On Windows: Use Git Credential Manager
   - On Mac/Linux: Token will be cached

#### Option B: SSH Key

```bash
# Generate SSH key
ssh-keygen -t ed25519 -C "your.email@example.com"
# Press Enter to accept default location
# Press Enter twice to skip passphrase (or set one for security)

# Copy the public key
cat ~/.ssh/id_ed25519.pub
# Copy the entire output

# Add to GitHub:
# GitHub → Settings → SSH and GPG keys → New SSH key
# Paste your key and save

# Update remote URL to SSH
git remote set-url origin git@github.com:yourusername/credit_scoring.git

# Try pushing again
git push -u origin main
```

---

## ✅ Step 10: Verify Upload

1. Go to your GitHub repository: `https://github.com/yourusername/credit_scoring`
2. Refresh the page
3. You should see all your files:
   - 📄 README.md (displayed on main page)
   - 🐍 Python scripts
   - 📊 CSV data files
   - 📈 Confusion matrix image
   - 📁 saved/ folder with models

---

## 🎨 Step 11: Enhance Your Repository

### Add a License

1. On GitHub repository page, click **Add file** → **Create new file**
2. Type filename: `LICENSE`
3. GitHub will show **Choose a license template** button
4. Select **MIT License** (or your preference)
5. Click **Commit new file**

### Add Topics (Tags)

1. On repository page, click ⚙️ next to "About"
2. Add topics: `machine-learning`, `credit-scoring`, `python`, `scikit-learn`, `random-forest`
3. Save changes

---

## 🔄 Making Future Updates

When you modify your code, use these commands:

```bash
# Check what changed
git status

# Add specific files
git add train.py
# Or add all changes
git add .

# Commit with descriptive message
git commit -m "Improve Random Forest accuracy with hyperparameter tuning"

# Push to GitHub
git push
```

### Example Workflow

```bash
# After improving your model
git add train.py evaluate.py
git commit -m "Add hyperparameter tuning to Random Forest"
git push

# After adding new feature
git add features.py
git commit -m "Add credit utilization ratio feature"
git push

# After updating documentation
git add README.md
git commit -m "Update README with latest performance metrics"
git push
```

---

## 🌿 Working with Branches

For new features, create branches:

```bash
# Create and switch to new branch
git checkout -b feature/xgboost-model

# Make changes, then commit
git add train.py
git commit -m "Add XGBoost model"

# Push branch to GitHub
git push -u origin feature/xgboost-model

# On GitHub, create a Pull Request to merge into main
```

---

## 🐛 Troubleshooting

### Problem: "remote origin already exists"
```bash
git remote remove origin
git remote add origin https://github.com/yourusername/credit_scoring.git
```

### Problem: "failed to push some refs"
```bash
# Pull latest changes first
git pull origin main --rebase

# Then push
git push -u origin main
```

### Problem: "Permission denied (publickey)"
Use HTTPS instead:
```bash
git remote set-url origin https://github.com/yourusername/credit_scoring.git
```

### Problem: Large file errors (models.pkl is ~700KB)
This should be fine, but if you get errors:
```bash
# Option 1: Remove from tracking
git rm --cached saved/models.pkl
echo "saved/*.pkl" >> .gitignore
git commit -m "Remove large model files from tracking"

# Option 2: Use Git LFS (Large File Storage)
git lfs install
git lfs track "*.pkl"
git add .gitattributes
git commit -m "Track pkl files with Git LFS"
```

### Problem: Merge conflicts
```bash
# Pull and manually resolve conflicts
git pull origin main

# Edit conflicting files, then:
git add .
git commit -m "Resolve merge conflicts"
git push
```

---

## 📊 Useful Git Commands

| Command | Description |
|---------|-------------|
| `git status` | Check current status and changes |
| `git log` | View commit history |
| `git log --oneline` | Compact commit history |
| `git diff` | See changes before committing |
| `git branch` | List all branches |
| `git checkout -b branch_name` | Create and switch to new branch |
| `git checkout main` | Switch back to main branch |
| `git merge branch_name` | Merge branch into current branch |
| `git pull` | Download latest changes from GitHub |
| `git clone URL` | Clone a repository |
| `git reset --hard HEAD` | Discard all local changes (⚠️ careful!) |

---

## 🎯 Best Practices

1. ✅ **Commit often** with clear, descriptive messages
2. ✅ **Don't commit sensitive data** (API keys, passwords, personal data)
3. ✅ **Use .gitignore** to exclude unnecessary files
4. ✅ **Write good commit messages**:
   - ❌ Bad: "fixed stuff"
   - ✅ Good: "Fix data preprocessing bug in features.py"
5. ✅ **Pull before pushing** to avoid conflicts
6. ✅ **Test your code** before committing
7. ✅ **Keep README updated** with accurate information

---

## 📚 Next Steps

### Enhance Your Project

- [ ] Add Jupyter notebooks for exploratory data analysis
- [ ] Create a Flask/FastAPI web service for predictions
- [ ] Add unit tests with pytest
- [ ] Set up GitHub Actions for CI/CD
- [ ] Create documentation with Sphinx
- [ ] Add model explainability (SHAP values)
- [ ] Deploy to Heroku/Railway/Streamlit Cloud

### GitHub Features to Explore

- **Issues**: Track bugs and feature requests
- **Projects**: Organize tasks with Kanban boards
- **Wiki**: Create detailed documentation
- **GitHub Actions**: Automate testing and deployment
- **Releases**: Tag and version your code
- **README badges**: Add status badges for build, coverage, etc.

---

## 🔗 Helpful Resources

- **Git Documentation**: https://git-scm.com/doc
- **GitHub Guides**: https://guides.github.com
- **Git Cheat Sheet**: https://education.github.com/git-cheat-sheet-education.pdf
- **Try Git Interactive Tutorial**: https://try.github.io

---

## 📧 Need Help?

If you encounter issues:
1. Check this guide's troubleshooting section
2. Search on Stack Overflow: https://stackoverflow.com/questions/tagged/git
3. GitHub Community Forum: https://github.community
4. Git official help: `git --help`

---

## ✨ Congratulations!

Your credit scoring project is now on GitHub! 🎉

Your repository is live at:
`https://github.com/yourusername/credit_scoring`

Share it with:
- Potential employers
- Collaborators
- The data science community

---

**Happy Coding! 🚀**
