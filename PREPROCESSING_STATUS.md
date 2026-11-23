# 🔄 Preprocessing Status

## Current Progress

**Status**: ✅ RUNNING

**Phase**: [1/6] Scanning Violence videos

**Progress**: ~58% (579/1000 videos scanned)

**Speed**: ~20-50 videos/second

**Estimated time for scanning**: ~1-2 minutes remaining

## Process Steps

### ✅ Step 1: Scanning Violence videos (In Progress)
- Checking video integrity
- Extracting metadata (frames, fps, duration)
- Current: 579/1000 videos

### ⏳ Step 2: Scanning Non-Violence videos (Pending)
- Will scan 1000 Non-Violence videos
- Same integrity checks

### ⏳ Step 3: Removing outliers (Pending)
- Statistical analysis
- Remove corrupted/invalid videos

### ⏳ Step 4: Loading dataset with Optical Flow (Pending)
- **This is the longest step** (~2-3 hours)
- Extracts RGB frames
- Calculates Optical Flow for each video
- Processes ~2000 videos

### ⏳ Step 5: Quality check (Pending)
- Validates data quality
- Checks for NaN/Inf values
- Analyzes movement intensity

### ⏳ Step 6: Saving data (Pending)
- Saves to `preprocessed_data_optical_flow.pkl`
- File size: ~2-4 GB

## Timeline

| Step | Duration | Status |
|------|----------|--------|
| 1. Scan Violence | ~1 min | 🔄 In Progress (58%) |
| 2. Scan Non-Violence | ~1 min | ⏳ Pending |
| 3. Remove outliers | <1 min | ⏳ Pending |
| 4. Extract RGB + Flow | 2-3 hours | ⏳ Pending |
| 5. Quality check | <1 min | ⏳ Pending |
| 6. Save data | ~1 min | ⏳ Pending |

**Total estimated time**: 2-3 hours

## Monitoring

To check progress:
```bash
# Check process output
Get-Process python

# View log file (if created)
tail -f preprocessing.log
```

## What's Happening Now

The script is:
1. Opening each video file
2. Checking if it can be read
3. Extracting metadata (frame count, FPS, resolution)
4. Testing if frames can be decoded
5. Marking videos as valid or corrupted

## Next Phase

After scanning completes, the script will:
1. Calculate Optical Flow for each video (this takes the most time)
2. Extract 30 frames per video
3. Compute motion vectors between consecutive frames
4. Save everything to a pickle file

## Output

When complete, you'll have:
- `preprocessed_data_optical_flow.pkl` (~2-4 GB)
- Contains:
  - RGB frames: (N, 30, 224, 224, 3)
  - Optical Flow: (N, 30, 224, 224, 2)
  - Labels: (N,)
  - Metadata

## After Preprocessing

Next step will be training:
```bash
python run_training.py
```

---

**Started**: Now
**Current step**: Scanning videos (1/6)
**Status**: Running smoothly ✅
