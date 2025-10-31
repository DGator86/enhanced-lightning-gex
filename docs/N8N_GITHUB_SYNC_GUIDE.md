# 🔗 N8N GitHub Sync - Complete Guide

## ✅ **You Have N8N Connected to GitHub - Perfect!**

This guide shows exactly how to import the Lightning GEX workflow using your N8N GitHub integration.

---

## 📁 **Repository Structure for N8N**

Create this structure in your GitHub repository:

```
your-repo/
├── workflows/
│   └── Lightning_GEX_Enhanced.json          # The workflow file
├── .n8n/
│   └── config.json                          # N8N settings (optional)
└── README.md
```

---

## 🚀 **Method 1: Direct GitHub Import** (RECOMMENDED)

### **Step 1: Push Workflow to Your GitHub**

```bash
# 1. Create workflows folder in your repo
mkdir -p workflows

# 2. Copy the workflow file
cp n8n_lightning_gex_workflow.json workflows/Lightning_GEX_Enhanced.json

# 3. Commit and push
git add workflows/Lightning_GEX_Enhanced.json
git commit -m "Add Enhanced Lightning GEX workflow"
git push origin main
```

### **Step 2: Pull in N8N**

1. **In N8N**: Go to **Settings** → **Version Control**
2. Click **"Pull from Remote"** or **"Sync"**
3. N8N fetches: `workflows/Lightning_GEX_Enhanced.json`
4. **Done!** Workflow appears in your workflows list

---

## 🔄 **Method 2: N8N Auto-Sync** (Best for Updates)

### **Configure Auto-Sync**

1. **In N8N**: Settings → Version Control
2. **Set Sync Interval**: 
   - Every 15 minutes
   - Every hour
   - Manual only
3. **Workflow Folder**: `workflows`
4. **Branch**: `main`

### **What Gets Synced**:
```
✅ workflows/*.json          → Auto-imported to N8N
✅ New workflows             → Auto-appear
✅ Workflow updates          → Auto-update
❌ Credentials               → Never synced (security)
❌ .env files                → Never synced (security)
```

---

## 📤 **Method 3: N8N API Push** (Programmatic)

If you have N8N API enabled:

```bash
# Get workflow content
WORKFLOW_JSON=$(cat workflows/Lightning_GEX_Enhanced.json)

# Push to N8N via API
curl -X POST "https://your-n8n-instance.com/api/v1/workflows" \
  -H "Content-Type: application/json" \
  -H "X-N8N-API-KEY: your-api-key" \
  -d "$WORKFLOW_JSON"
```

---

## 🗂️ **GitHub Repository Setup**

### **Option A: Create New Repo**

```bash
# 1. Create repo on GitHub
# Repository name: enhanced-lightning-gex

# 2. Initialize locally
mkdir enhanced-lightning-gex
cd enhanced-lightning-gex
git init

# 3. Create structure
mkdir -p workflows config backend docs

# 4. Copy files
cp /path/to/n8n_lightning_gex_workflow.json workflows/Lightning_GEX_Enhanced.json
cp /path/to/.env.example config/
cp /path/to/docker-compose.yml config/
# ... copy other files

# 5. Push to GitHub
git add .
git commit -m "Initial commit: Enhanced Lightning GEX"
git remote add origin https://github.com/your-username/enhanced-lightning-gex.git
git push -u origin main
```

### **Option B: Add to Existing Repo**

```bash
# If you already have a repo
cd your-existing-repo

# Create workflows folder
mkdir -p workflows

# Copy workflow
cp /path/to/n8n_lightning_gex_workflow.json workflows/Lightning_GEX_Enhanced.json

# Commit
git add workflows/Lightning_GEX_Enhanced.json
git commit -m "Add Lightning GEX workflow"
git push
```

---

## ⚙️ **N8N Version Control Settings**

### **Recommended Configuration**

```json
{
  "versionControl": {
    "connected": true,
    "repositoryUrl": "https://github.com/your-username/enhanced-lightning-gex",
    "branch": "main",
    "workflowsFolder": "workflows",
    "autoSync": true,
    "syncInterval": 900,  // 15 minutes
    "authorName": "Your Name",
    "authorEmail": "your-email@example.com"
  }
}
```

### **Apply in N8N**:
1. Settings → Version Control
2. Configure:
   - **Repository**: Your GitHub repo URL
   - **Branch**: `main`
   - **Workflows Folder**: `workflows`
   - **Auto-Sync**: Enabled
   - **Interval**: 15 minutes

---

## 🔐 **Credentials Management**

### **Important**: Credentials are NEVER synced to GitHub

**Credentials to add in N8N UI:**
- ✅ Alpaca API (Key + Secret)
- ✅ Telegram (Bot Token + Chat ID) - optional
- ✅ Email SMTP - optional
- ✅ Discord Webhook - optional

**After importing workflow:**
1. Click on any node with credentials
2. Add new credential
3. Enter your keys
4. Test connection
5. Save

**Credentials are stored locally in N8N, not in GitHub** ✅

---

## 🎯 **Import Verification**

### **Check Import Success**

1. **In N8N**: Go to **Workflows**
2. Look for: **"Lightning GEX - Enhanced Trading System"**
3. Open it
4. Verify:
   - ✅ 24 nodes visible
   - ✅ All connections intact
   - ✅ Schedule trigger configured
   - ✅ No missing nodes

### **Node Count Check**:
```
Expected nodes: 24
- 1 Schedule Trigger
- 1 Market Hours Check
- 1 Load Ticker List
- 1 Fetch Market Data
- 1 Prepare Analysis Request
- 1 Enhanced Lightning Analysis
- 1 Trade Approval Filter
- 1 Format Order
- 1 Place Order
- 1 Format Alert
- 3 Alert nodes (Telegram, Email, Discord)
- 1 Log to Database
- 1 Monitor Open Positions
- 1 Exit Signal Check
- 1 Close Position
- 1 Update Trade Outcome
- 1 Hourly Health Check
- 1 Get System Status
- 1 Check Circuit Breaker
- 1 Circuit Breaker Alert
```

---

## 🔄 **Keeping Workflow Updated**

### **When I Update the Workflow**:

1. **I push to GitHub**:
   ```bash
   git add workflows/Lightning_GEX_Enhanced.json
   git commit -m "Update: Improved multi-agent logic"
   git push
   ```

2. **Your N8N auto-pulls** (if auto-sync enabled)
   - Or manually: Settings → Version Control → Pull

3. **Workflow updates automatically** ✅

### **Version History**:
```
✅ All changes tracked in Git
✅ Can revert to any previous version
✅ See what changed in each update
```

---

## 📋 **Complete Setup Checklist**

### **GitHub Setup**:
- [ ] Repository created
- [ ] Workflow file pushed to `workflows/` folder
- [ ] Repository URL copied

### **N8N Setup**:
- [ ] Version Control connected to GitHub
- [ ] Repository URL configured
- [ ] Branch set to `main`
- [ ] Workflows folder set to `workflows`
- [ ] Auto-sync enabled (optional)
- [ ] First sync completed

### **Workflow Import**:
- [ ] Workflow appears in N8N
- [ ] All 24 nodes visible
- [ ] Connections intact
- [ ] No errors shown

### **Credentials**:
- [ ] Alpaca API added
- [ ] Alert channels configured (optional)
- [ ] All credentials tested

### **Activation**:
- [ ] Workflow activated (toggle ON)
- [ ] First execution successful
- [ ] Alerts working (if configured)

---

## 🐛 **Troubleshooting**

### **"N8N can't find workflow"**

```bash
# Check file location
ls -la workflows/

# File should be:
workflows/Lightning_GEX_Enhanced.json

# NOT:
workflows/subfolder/Lightning_GEX_Enhanced.json  # Wrong!
```

### **"Pull failed" error**

1. Check repository URL is correct
2. Verify branch exists (`main` or `master`)
3. Check GitHub access token has read permissions
4. Try manual pull: Settings → Version Control → Pull

### **"Workflow imported but has errors"**

- Some credentials missing (expected)
- Add credentials in each node that needs them
- HTTP Request nodes: Add authentication
- Alert nodes: Add API keys

### **"Auto-sync not working"**

1. Check auto-sync is enabled
2. Verify sync interval is set
3. Check N8N logs for errors:
   ```bash
   docker logs n8n --tail 100 | grep version
   ```

---

## 💡 **Pro Tips**

### **Workflow Naming**:
```
✅ Good: Lightning_GEX_Enhanced.json
✅ Good: lightning-gex-trading.json
❌ Bad: workflow (123).json
❌ Bad: my workflow.json  (spaces)
```

### **Multiple Workflows**:
```
workflows/
├── Lightning_GEX_Enhanced.json        # Main trading workflow
├── Lightning_GEX_Monitoring.json      # Monitoring workflow
└── Lightning_GEX_Backtesting.json     # Backtesting workflow
```

### **Branches**:
```
main          → Production workflow (stable)
development   → Testing new features
experimental  → Trying new strategies
```

---

## 📞 **Need Help?**

### **Quick Test**:
```bash
# 1. Check if file is in GitHub
curl https://raw.githubusercontent.com/your-username/your-repo/main/workflows/Lightning_GEX_Enhanced.json

# 2. Should return JSON starting with:
{
  "name": "Lightning GEX - Enhanced Trading System",
  "nodes": [...]
}
```

### **Common Issues**:
1. **File not found**: Check path is `workflows/Lightning_GEX_Enhanced.json`
2. **Invalid JSON**: Validate at jsonlint.com
3. **N8N not pulling**: Check Settings → Version Control → Status
4. **Credentials missing**: Add manually in N8N UI

---

## 🎉 **You're Ready!**

**Your N8N is connected to GitHub, so importing is automatic:**

1. ✅ Push workflow to `workflows/` folder in your repo
2. ✅ N8N auto-pulls the workflow
3. ✅ Add credentials in N8N UI
4. ✅ Activate and start trading!

**Workflow will stay in sync with your GitHub repo automatically.** 🚀
