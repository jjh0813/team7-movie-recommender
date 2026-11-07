param([Parameter(Mandatory=$true)][string]$SourceUrl,
      [Parameter(Mandatory=$true)][string]$Destination)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
Invoke-WebRequest -Uri $SourceUrl -OutFile $Destination
