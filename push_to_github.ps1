# Script to push code to GitHub
# Run this after creating the repository at https://github.com/Capgemini-Innersource/ptr_LangGraph_WatsonX

# Stage all files
git add .

# Commit
git commit -m "Initial commit: LangGraph ReAct Agent with FastAPI and React UI"

# Add remote (update this URL after creating the repo)
git remote add origin https://github.com/Capgemini-Innersource/ptr_LangGraph_WatsonX.git

# Push to main branch
git branch -M main
git push -u origin main

Write-Host "Code pushed to GitHub successfully!"
Write-Host "Repository: https://github.com/Capgemini-Innersource/ptr_LangGraph_WatsonX"
