# Installation Guide

Welcome! Before starting, please make sure you install some basic things that we need throughout this course.

## What you'll install:

- VS Code (code editor)

- Git

- Configure SSH Key access in Github

- Python

- Creating a virtual environment, activating the environment and installing some dependencies

- Run a jupyter notebook `.ipynb file` in VScode

<br>

---

## Installing Visual Studio Code

1. Go to [code.visualstudio.com](https://code.visualstudio.com/download)

2. Download and open the downloaded file and follow the instructions to install VS Code

3. Open VS Code after installation

**Important:** Make sure to check the "Add to PATH" option during installation

### Recommended VS Code Extensions

After installing VS Code, install these helpful extensions:

- **Python** (by Microsoft) - Essential for Python development

- **Ruff** (by Astral Software) - Python linter and code formatter

- **Jupyter** (by Microsoft) - For running Jupyter notebooks

- **Git Graph** - Visual representation of your Git repositories

- **GitLens** - Enhanced Git capabilities

To install extensions:

1. Open VS Code

2. Click the Extensions icon in the sidebar (or press `Ctrl+Shift+X`)

3. Search for each extension and click "Install"

<br>

---

## Setting up Git

1. Go to [git-scm.com](https://git-scm.com/downloads)

2. Download and install Git for your operating system (Windows, Mac, or Linux)

3. Open a terminal or command prompt and type:

   ```bash
   git --version
   ```
   If you see a version number, Git is installed!

*Tip: On Windows, you can use Git Bash (installed with Git) for a better terminal experience.*

*Make sure to check the necessary boxes while installing git*

## Installing Visual Studio Code

1. Go to [code.visualstudio.com](https://code.visualstudio.com/download)

2. Download and open the downloaded file and follow the instructions to install VS Code

3. Open VS Code after installation

**Important:** Make sure to check the "Add to PATH" option during installation

### Recommended VS Code Extensions

After installing VS Code, install these helpful extensions:
- **Python** (by Microsoft) - Essential for Python development

- **Jupyter** (by Microsoft) - For running Jupyter notebooks

- **Git Graph** - Visual representation of your Git repositories

- **Ruff** - Python extension for linting and formatting

- **GitLens** - Enhanced Git capabilities

To install extensions:
1. Open VS Code

2. Click the Extensions icon in the sidebar (or press `Ctrl+Shift+X`)

3. Search for each extension and click "Install"

---


## Setting up Git

### Installation

1. Go to [git-scm.com](https://git-scm.com/downloads)

2. Download and install Git for your operating system (Windows, Mac, or Linux)

3. **Important installation options:**

   - ✅ Check "Add Git to PATH"

   - ✅ Use Git from the command line and 3rd-party software

   - ✅ Use the OpenSSL library

   - ✅ Checkout Windows-style, commit Unix-style line endings

4. Open a terminal or command prompt and type:

   ```bash
   git --version
   ```

   If you see a version number, Git is installed successfully!

### Configure Git (First-time setup)

Set up your identity for Git commits:

```bash
git config --global user.name "Your Github Username"
git config --global user.email "github.email@example.com"
```

**Note:** Use the same email address as your GitHub account.

**Tip:** On Windows, you can use Git Bash (installed with Git) for a better terminal experience.

---

## Configure SSH Key Access in GitHub

### Why SSH Keys?

SSH keys provide a secure way to connect to GitHub without entering your username and password every time.

### Step 1: Check for existing SSH keys

Open terminal/Git Bash and run:

```bash
ls -al ~/.ssh
```

If you see files like `id_rsa` and `id_rsa.pub` or `id_ed25519` and `id_ed25519.pub`, you already have SSH keys.

### Step 2: Generate a new SSH key (if needed)

```bash
ssh-keygen -t ed25519 -C "your.email@example.com"
```

When prompted:

- Press Enter to save in the default location

- Enter a passphrase (optional but recommended)


### Step 3: Add SSH key to GitHub

1. Copy your public key:
   ```bash
   cat ~/.ssh/id_ed25519.pub
   ```

2. Go to [GitHub.com](https://github.com) and log in

3. Click your profile picture → Settings

4. In the sidebar, click "SSH and GPG keys"

5. Click "New SSH key"

6. Give it a title (e.g., "My Laptop")

7. Paste your key in the "Key" field

8. Click "Add SSH key"

### Step 5: Test your connection

```bash
ssh -T git@github.com
```

You should see a message like: "Hi username! You've successfully authenticated..."

---


## Installing Python

### Option 1: Official Python Installer (Recommended for beginners)

1. Go to [python.org/downloads](https://www.python.org/downloads/)

2. Download Python 3.11 or later

3. **Important:** During installation, check "Add Python to PATH"

4. Verify installation:

   ```bash
   python --version
   ```
   or
   ```bash
   python3 --version
   ```

### Option 2: Using Package Managers

**Windows (using winget):**
```bash
winget install Python.Python.3.11
```

**Mac (using Homebrew):**
```bash
brew install python@3
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install python3 python3-venv python3-pip
```

### Verify pip installation

```bash
pip --version
```

If pip is not found, install it:
```bash
python -m ensurepip --upgrade
```

---

## Creating and Managing Virtual Environments

### Why Virtual Environments?

Virtual environments keep your project dependencies isolated and prevent conflicts between different projects.

### Step 1: Create a virtual environment

Navigate to your project folder and run:

```bash
python -m venv myenv
```

*For any issues in powershell you can execute this line*

On Microsoft Windows, it may required to enable the Activate.ps1 script by setting the execution policy for the user. You can do this by issuing the following Powershell command:

```bash
PS C:\> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Replace `myenv` with your preferred environment name.

### Step 2: Activate the virtual environment

**Windows:**

```bash
myenv\Scripts\activate
```

Using git bash

```bash
source myenv/Scripts/activate
```

**Mac/Linux:**

```bash
source myenv/bin/activate
```

**Success indicator:** Your terminal prompt should show `(myenv)` at the beginning.

### Step 3: Install dependencies

With your virtual environment activated, install packages:

```bash
# Essential data science packages
pip install uv

uv pip install ipykernel pandas numpy matplotlib seaborn statsmodels scikit-learn

# Or install from a requirements file
pip install -r requirements.txt
```

### Step 4: Deactivate when done

```bash
deactivate
```

### Creating a requirements.txt file

To save your current dependencies:
```bash
pip freeze > requirements.txt
```

---


## Common Issues and Solutions

### Git Issues
- **Problem:** Git commands not recognized

- **Solution:** Reinstall Git and ensure "Add to PATH" is checked

### Python Issues
- **Problem:** Python not found

- **Solution:** Add Python to system PATH or reinstall with PATH option checked

### Virtual Environment Issues
- **Problem:** Cannot activate virtual environment

- **Solution:** Check the activation command for your OS and ensure you're in the correct directory

### Jupyter Issues
- **Problem:** Kernel not starting in VS Code

- **Solution:** Install ipykernel in your virtual environment and restart VS Code

---

**Great work! You’ve successfully completed this tutorial.**

Before moving on to the next one, double-check that you’ve followed all the steps carefully.

**✨ Happy coding, and see you in the next tutorial!**
