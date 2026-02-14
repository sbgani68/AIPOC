# AVD Capacity Remediation Script
# Generated: 2026-02-14T22:40:43.845249
# Severity: High

# Azure CLI commands to scale host pool

$ResourceGroup = "YOUR_RESOURCE_GROUP"
$HostPoolName = "YOUR_HOST_POOL"

Write-Host "Scaling AVD Host Pool..." -ForegroundColor Yellow

# Check current capacity
az desktopvirtualization hostpool show `
    --resource-group $ResourceGroup `
    --name $HostPoolName

# Recommended Actions:
# - Validate host pool session distribution and drain overloaded hosts.
# - Review recent gateway/session connection failures and retry trends.
# - Check FSLogix/profile and disconnection events in corresponding timeframe.

# Example: Add new session hosts
# Update max session limit if needed
az desktopvirtualization hostpool update `
    --resource-group $ResourceGroup `
    --name $HostPoolName `
    --max-session-limit 20

Write-Host "✓ Remediation complete" -ForegroundColor Green
