# FPS Control Code Report

## Issue: FPS control doesn't work

## Implementation Overview

The FPS control should flow:
1. User selects FPS in Settings
2. `waterfallFpsProvider` state changes
3. `video_waterfall_display.dart` listens for change
4. Sends `set_fps` command via `video_stream_provider.dart`
5. Backend receives command and updates `server.video_fps`
6. Backend loop uses updated FPS for frame interval

---

## 1. Settings Screen - Provider Definition

**File:** `lib/features/settings/settings_screen.dart`

```dart
/// Waterfall FPS setting (frames per second)
/// Controls how fast the waterfall streams
/// Default 30fps - full speed
final waterfallFpsProvider = StateProvider<int>((ref) => 30);
```

**FPS Selector Widget:**
```dart
/// Waterfall FPS selector - controls streaming speed for debugging
class _WaterfallFpsSelector extends ConsumerWidget {
  const _WaterfallFpsSelector();

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final fps = ref.watch(waterfallFpsProvider);

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            const Text(
              'Stream FPS',
              style: TextStyle(
                fontSize: 14,
                color: G20Colors.textPrimaryDark,
              ),
            ),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
              decoration: BoxDecoration(
                color: fps < 30 
                    ? Colors.orange.withValues(alpha: 0.2)
                    : G20Colors.primary.withValues(alpha: 0.2),
                borderRadius: BorderRadius.circular(4),
              ),
              child: Text(
                '${fps}fps',
                style: TextStyle(
                  fontWeight: FontWeight.bold,
                  color: fps < 30 ? Colors.orange : G20Colors.primary,
                ),
              ),
            ),
          ],
        ),
        const SizedBox(height: 4),
        Text(
          'Slow down waterfall for debugging (affects data rate)',
          style: TextStyle(
            fontSize: 11,
            color: G20Colors.textSecondaryDark,
          ),
        ),
        const SizedBox(height: 12),
        Row(
          children: [
            _FpsOption(
              label: '1',
              value: 1,
              selected: fps == 1,
              onTap: () => ref.read(waterfallFpsProvider.notifier).state = 1,
            ),
            const SizedBox(width: 6),
            _FpsOption(
              label: '5',
              value: 5,
              selected: fps == 5,
              onTap: () => ref.read(waterfallFpsProvider.notifier).state = 5,
            ),
            const SizedBox(width: 6),
            _FpsOption(
              label: '10',
              value: 10,
              selected: fps == 10,
              onTap: () => ref.read(waterfallFpsProvider.notifier).state = 10,
            ),
            const SizedBox(width: 6),
            _FpsOption(
              label: '15',
              value: 15,
              selected: fps == 15,
              onTap: () => ref.read(waterfallFpsProvider.notifier).state = 15,
            ),
            const SizedBox(width: 6),
            _FpsOption(
              label: '30',
              value: 30,
              selected: fps == 30,
              onTap: () => ref.read(waterfallFpsProvider.notifier).state = 30,
            ),
          ],
        ),
      ],
    );
  }
}
```

---

## 2. Video Stream Provider - setFps Method

**File:** `lib/features/live_detection/providers/video_stream_provider.dart`

```dart
void setFps(int fps) {
  debugPrint('[VideoStream] setFps: $fps');
  
  if (_channel == null) {
    debugPrint('[VideoStream] Cannot set FPS - not connected');
    return;
  }
  
  final msg = json.encode({
    'command': 'set_fps',
    'fps': fps,
  });
  
  try {
    _channel!.sink.add(msg);
    debugPrint('[VideoStream] Sent FPS command: ${fps}fps');
  } catch (e) {
    debugPrint('[VideoStream] Send FAILED: $e');
  }
}
```

---

## 3. Video Waterfall Display - Listener

**File:** `lib/features/live_detection/widgets/video_waterfall_display.dart`

**Import:**
```dart
import '../../settings/settings_screen.dart' show waterfallTimeSpanProvider, waterfallFpsProvider;
```

**Listener in build() method:**
```dart
// Listen for FPS changes and send to backend
ref.listen<int>(waterfallFpsProvider, (previous, next) {
  final currentState = ref.read(videoStreamProvider);
  debugPrint('[Waterfall] FPS listener fired: $previous → $next, connected: ${currentState.isConnected}');
  if (previous != next && currentState.isConnected) {
    ref.read(videoStreamProvider.notifier).setFps(next);
  }
});

// Send initial time span and FPS when connection state changes to connected
ref.listen<VideoStreamState>(videoStreamProvider, (previous, next) {
  if (previous?.isConnected != true && next.isConnected) {
    final currentTimeSpan = ref.read(waterfallTimeSpanProvider);
    if ((currentTimeSpan - 5.0).abs() > 0.1) {
      ref.read(videoStreamProvider.notifier).setTimeSpan(currentTimeSpan);
    }
    final currentFps = ref.read(waterfallFpsProvider);
    if (currentFps != 30) {
      ref.read(videoStreamProvider.notifier).setFps(currentFps);
    }
  }
});
```

---

## 4. Backend - Command Handler

**File:** `backend/unified_pipeline.py`

**In `video_ws_handler()` function:**
```python
elif cmd == 'set_fps':
    try:
        new_fps = int(data.get('fps', 30))
        new_fps = max(1, min(60, new_fps))  # Clamp to 1-60
        
        old_fps = server.video_fps
        server.video_fps = new_fps
        
        # Recalculate suggested buffer height based on new FPS
        new_suggested_height = int(server.time_span_seconds * new_fps * server.rows_per_frame)
        server.suggested_buffer_height = new_suggested_height
        
        print(f"[Pipeline] FPS changing: {old_fps} -> {new_fps}fps (suggested buffer: {new_suggested_height} rows)", flush=True)
        
        # Send updated metadata to client
        metadata = {
            'type': 'metadata',
            'mode': 'row_strip',
            'strip_width': server.video_width,
            'rows_per_strip': server.rows_per_frame,
            'video_fps': new_fps,
            'suggested_buffer_height': new_suggested_height,
            'time_span_seconds': server.time_span_seconds,
            'encoder': 'rgba_raw',
        }
        await websocket.send(bytes([server.MSG_METADATA]) + json.dumps(metadata).encode())
        
        # Send acknowledgment
        await websocket.send(json.dumps({
            'type': 'fps_ack',
            'fps': new_fps,
            'suggested_buffer_height': new_suggested_height,
        }))
        print(f"[Pipeline] FPS change complete!", flush=True)
        
    except Exception as e:
        print(f"[Pipeline] ERROR in set_fps: {e}", flush=True)
        import traceback
        traceback.print_exc()
```

---

## 5. Backend - Dynamic FPS in Pipeline Loop

**File:** `backend/unified_pipeline.py`

**In `run_pipeline()` method:**
```python
async def run_pipeline(self, websocket):
    """Main row-strip streaming loop - sends strips instead of full frames."""
    import zlib
    
    self.is_running = True
    frame_count = 0
    
    # Send metadata first
    await self.send_metadata(websocket)
    
    logger.info(f"Row-strip pipeline started ({self.video_fps}fps, ~{self.rows_per_frame} rows/frame)")
    
    while self.is_running:
        try:
            frame_start = time.perf_counter()
            
            # Dynamic FPS - read from self.video_fps each iteration
            frame_interval = 1.0 / self.video_fps
            
            # ... rest of frame processing ...
            
            # Rate limit
            elapsed = time.perf_counter() - frame_start
            sleep_time = max(0.001, frame_interval - elapsed)
            await asyncio.sleep(sleep_time)
```

---

## Debugging Steps

### 1. Check if Settings UI is working
- Go to Settings
- Click on FPS options (1, 5, 10, 15, 30)
- Verify the selected button highlights
- **Expected debug output:** `[Waterfall] FPS listener fired: 30 → 5, connected: true`

### 2. Check if WebSocket command is sent
- **Expected debug output:** `[VideoStream] setFps: 5`
- **Expected debug output:** `[VideoStream] Sent FPS command: 5fps`

### 3. Check if Backend receives command
- **Expected debug output in backend terminal:**
  ```
  [WS RECV] *** MESSAGE RECEIVED ***
  [WS RECV] Type: <class 'str'>
  [WS RECV] Text message: {"command":"set_fps","fps":5}
  [WS RECV] Command: set_fps
  [Pipeline] FPS changing: 30 -> 5fps (suggested buffer: 950 rows)
  [Pipeline] FPS change complete!
  ```

### 4. Check if frame rate actually changes
- At 30fps: ~33ms between frames
- At 5fps: ~200ms between frames
- Watch the waterfall - it should visibly slow down

---

## Potential Issues

### Issue 1: Listener not in scope
The FPS listener is in `video_waterfall_display.dart`, but the Settings screen is a different route. The listener only fires when the waterfall is visible.

**Fix:** The listener should work because `ref.listen` persists during the widget lifecycle, and changing settings should update the provider globally.

### Issue 2: Provider not exported
The `waterfallFpsProvider` needs to be exported from `settings_screen.dart`.

**Verify:** The import statement exists:
```dart
import '../../settings/settings_screen.dart' show waterfallTimeSpanProvider, waterfallFpsProvider;
```

### Issue 3: Message not decoded properly
The backend might not be receiving the message. Check if `video_ws_handler` is actually receiving messages.

### Issue 4: Waterfall display not mounted
If the user goes to Settings and changes FPS, the waterfall widget might not be mounted, so the listener doesn't fire.

**Fix:** Move the FPS listener to `live_detection_screen.dart` which is always mounted, or to a provider that runs regardless of which tab is active.

---

## Recommended Fix

Move the FPS listener from `video_waterfall_display.dart` to `live_detection_screen.dart`:

```dart
// In _LiveDetectionScreenState.build()
ref.listen<int>(waterfallFpsProvider, (previous, next) {
  final currentState = ref.read(videoStreamProvider);
  debugPrint('[LiveDetection] FPS listener fired: $previous → $next, connected: ${currentState.isConnected}');
  if (previous != next && currentState.isConnected) {
    ref.read(videoStreamProvider.notifier).setFps(next);
  }
});
```

This ensures the listener is active as long as the Live Detection screen is visible, not just when the waterfall widget is visible.
