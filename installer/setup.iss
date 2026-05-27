#define AppVersion "1.0.0"

[Setup]
AppName=Seismic Visualizer
AppVersion={#AppVersion}
AppPublisher=Valerii Zghurovskyi
AppPublisherURL=https://github.com/valerii-zghurovskyi/seismic-visualizer
DefaultDirName={autopf}\SeismicVisualizer
DefaultGroupName=Seismic Visualizer
OutputDir=output
OutputBaseFilename=SeismicVisualizer_Setup_{#AppVersion}
SetupIconFile=..\assets\icon.ico
UninstallDisplayIcon={app}\SeismicVisualizer.exe
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Files]
Source: "..\dist\SeismicVisualizer\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Seismic Visualizer"; Filename: "{app}\SeismicVisualizer.exe"
Name: "{group}\Uninstall Seismic Visualizer"; Filename: "{uninstallexe}"
Name: "{commondesktop}\Seismic Visualizer"; Filename: "{app}\SeismicVisualizer.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional icons:"

[Run]
Filename: "{app}\SeismicVisualizer.exe"; Description: "&Launch Seismic Visualizer"; Flags: nowait postinstall skipifsilent
