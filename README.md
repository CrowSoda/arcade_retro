# G20 RF Detection Platform - Demo Client

A Flutter desktop application for the G20 RF detection platform. This client displays real-time waterfall spectrograms, detection boxes, and provides controls for RF parameters and model training.

## Features

### Live Detection View
- **Waterfall Display**: Real-time scrolling spectrogram with color-mapped power levels
- **PSD Chart**: Power Spectral Density graph showing current spectrum
- **Detection Overlays**: Color-coded bounding boxes for detected signals
- **Detection Table**: List of all active detections with class, confidence, and timestamps
- **Inputs Panel**: RF controls (frequency, gain, bandwidth) with preset buttons

### Training View
- **Spectrogram Review**: Full-width display for reviewing captured data
- **File Selector**: Browse and select IQ capture files
- **Data Warnings**: Alerts when insufficient training samples
- **Training Controls**: Epochs, learning rate, batch size inputs
- **Class Statistics**: Table showing sample counts per class

### Settings
- **Connection**: Host, gRPC port, UDP port configuration
- **Display**: Waterfall history, dB range, colormap selection
- **Model**: Active model info and selection
- **About**: Version info and documentation links

## Prerequisites

1. **Flutter SDK** (3.24+)
2. **Windows Developer Mode** enabled (required for Windows builds)
   - Run: `start ms-settings:developers` and enable Developer Mode
3. **Visual Studio 2022** with C++ desktop development workload (for Windows builds)

## Getting Started

### Install Dependencies
```bash
flutter pub get
```

### Run in Development Mode
```bash
flutter run -d windows
```

### Build Release
```bash
flutter build windows --release
```

The built executable will be at:
`build/windows/x64/runner/Release/g20_demo.exe`

## Project Structure

```
g20_demo/
├── lib/
│   ├── main.dart                    # App entry point
│   ├── app.dart                     # Root app widget
│   ├── core/
│   │   └── config/
│   │       ├── theme.dart           # Color palette & themes
│   │       └── router.dart          # Navigation routes
│   └── features/
│       ├── shell/
│       │   └── app_shell.dart       # Main layout with NavigationRail
│       ├── live_detection/
│       │   ├── live_detection_screen.dart
│       │   ├── widgets/
│       │   │   ├── waterfall_display.dart
│       │   │   ├── psd_chart.dart
│       │   │   ├── detection_table.dart
│       │   │   └── inputs_panel.dart
│       │   └── providers/
│       │       ├── waterfall_provider.dart
│       │       └── detection_provider.dart
│       ├── training/
│       │   └── training_screen.dart
│       └── settings/
│           └── settings_screen.dart
└── test/
    └── widget_test.dart
```

## Demo Mode

The app runs in demo mode by default, simulating:
- Waterfall data with noise floor and signals
- Detection boxes that appear, move, and disappear
- System state indicators

## Connecting to Real Backend

To connect to the G20 backend:

1. Update connection settings in the Settings screen
2. The app expects:
   - **gRPC server** on port 50051 for control commands
   - **UDP stream** on port 5000 for telemetry/waterfall data

## Next Steps

- [ ] Define proto files for gRPC communication
- [ ] Implement real UDP telemetry client
- [ ] Add mock server for development testing
- [ ] Connect to actual G20 hardware

## License

Proprietary - Internal Use Only
