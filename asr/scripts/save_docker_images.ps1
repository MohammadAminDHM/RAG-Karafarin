param(
  [string]$OutDir = "$(Resolve-Path (Join-Path $PSScriptRoot '..\\docker-images'))",
  [string]$WebImage = "stt-web:latest",
  [string]$DownloadModelImage = "stt-download_model:latest"
)

$ErrorActionPreference = "Stop"

Write-Host "Output directory: $OutDir"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

function Save-DockerImageTar {
  param(
    [Parameter(Mandatory=$true)][string]$Image,
    [Parameter(Mandatory=$true)][string]$OutFile
  )
  Write-Host "Saving Docker image '$Image' -> '$OutFile' ..."
  docker image inspect $Image *> $null
  docker image save -o $OutFile $Image
}

$webOut = Join-Path $OutDir "stt-web.tar"
$dmOut  = Join-Path $OutDir "stt-download_model.tar"

Save-DockerImageTar -Image $WebImage -OutFile $webOut
Save-DockerImageTar -Image $DownloadModelImage -OutFile $dmOut

Write-Host "`nDone. Exported:"
Get-ChildItem $OutDir -Filter "*.tar" | Format-Table Name,Length,LastWriteTime

