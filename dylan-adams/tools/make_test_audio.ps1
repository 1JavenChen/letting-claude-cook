# Generate a test speech clip with Windows' built-in text-to-speech.
# Usage: powershell -File tools/make_test_audio.ps1 [-Text "..."] [-Out data/test.wav]
param(
    [string]$Text = "Hello! Welcome to our virtual meeting. Please, put the blue boxes over by the window, and pass me the cheese.",
    [string]$Out = "data/test.wav"
)
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$format = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
$synth.SetOutputToWaveFile((Resolve-Path -LiteralPath (Split-Path $Out -Parent)).Path + "\" + (Split-Path $Out -Leaf), $format)
$synth.Speak($Text)
$synth.Dispose()
Write-Output "Wrote $Out"
