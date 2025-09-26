# Enhanced BEVFormer → Memory Bridge Integration Plan

## Current Phase

**Phase 4: Integration Testing & Validation 🧪**

---

## Implementation Status

- **✅ Phase 1: BEV Memory Feature Extractor**
  - `semantic-map/bev_memory_bridge.py` — Created
  - Enhanced `semantic-map/bevformer_integration.py` — Updated

- **✅ Phase 2: Missing Integration Components**
  - `integration/itri_uniad_inference.py` — Created
  - `integration/itri_data_adapter.py` — Created

- **✅ Phase 3: Memory Bridge Architecture**
  - BEV encoder output processing — Implemented
  - `args_tuple` population with memory features — Implemented

- **🔄 Phase 4: Integration Testing & Validation (In Progress)**
  - `integration/test_integration.py` — Created

---

## Phase 1: BEV Memory Feature Extractor ⭐ **Critical Priority**

### 1.1 Memory Feature Bridge (`semantic-map/bev_memory_bridge.py`)
- **Extract BEV Memory Components:**  
  From BEVFormer output, extract `memory`, `memory_mask`, `memory_pos` tensors.
- **Replace `args_tuple` None Values:**  
  Currently lines 115-118 in `motionformer_integration.py` have `None` placeholders.
- **Handle Tensor Compatibility:**  
  Ensure proper shapes and device placement for GPU/CPU operations.
- **Memory Positioning:**  
  Generate spatial positional encodings for BEV memory features.

### 1.2 Enhance BEVFormer Integration (`semantic-map/bevformer_integration.py`)
- **ITRI 4-Camera Processing:**  
  Leverage existing `BEVFormerDataProcessor` for multi-camera input.
- **CAN Bus Integration:**  
  Apply 18-dimensional CAN bus data from `canbus/*.pkl` files for temporal consistency.
- **BEV Feature Generation:**  
  Process images through BEVFormer encoder to generate `bev_embed` features.

---

## Phase 2: Missing Integration Components 🆕 **Newly Identified**

### 2.1 UniAD Integration Interface (`integration/itri_uniad_inference.py`)
- **Complete Pipeline Orchestrator:**  
  Bridge all 3 modules (semantic-map, track, BEVFormer) with UniAD architecture.
- **Custom `forward_test` Override:**  
  Replace standard UniAD pipeline with ITRI data flow.
- **Error Handling:**  
  Robust error handling for missing data or component failures.

### 2.2 ITRI Data Format Adapter (`integration/itri_data_adapter.py`)
- **Data Format Conversion:**  
  Convert ITRI formats → UniAD expected tensor formats.
- **Coordinate System Alignment:**  
  Ensure BEV coordinates match between semantic map, tracking, and BEV features.
- **Metadata Management:**  
  Handle timestamps, ego poses, and calibration parameters.

---

## Phase 3: Memory Bridge Architecture 🔧 **Technical Core**

### 3.1 BEV Encoder Output Processing
- **Memory Extraction:**  
  Extract intermediate features from BEVFormerEncoder layers.
- **Spatial Encoding:**  
  Generate position encodings matching BEV grid (200x200 from config).
- **Memory Masking:**  
  Create attention masks for valid BEV regions.

### 3.2 `args_tuple` Population

```python
args_tuple = [
    bev_memory,        # ← EXTRACT from BEVFormer (currently None)
    bev_memory_mask,   # ← GENERATE from valid regions (currently None)
    bev_memory_pos,    # ← SPATIAL position encodings (currently None)
    lane_query,        # ✅ Already working from semantic-map
    None,              # Unused slot
    lane_query_pos,    # ✅ Already working from semantic-map
    hw_lvl             # ✅ Already set [(200, 200)]
]
```

---

## Phase 4: Integration Testing & Validation 🧪

### 4.1 Component Integration Testing
- **Memory Feature Validation:**  
  Verify tensor shapes match MotionFormer expectations.
- **End-to-End Pipeline:**  
  Test complete ITRI → UniAD → Motion Prediction flow.
- **Performance Benchmarking:**  
  Ensure real-time inference capability.

### 4.2 Data Flow Validation
- **Memory-Motion Bridge:**  
  Confirm BEV memory features properly inform motion predictions.
- **Coordinate Consistency:**  
  Validate all coordinate systems align correctly.
- **GPU Memory Management:**  
  Optimize memory usage for large tensor operations.

---

## Deliverables

1. `bev_memory_bridge.py` — Memory feature extraction and `args_tuple` population
2. Enhanced `bevformer_integration.py` — Complete 4-camera + CAN bus processing
3. `itri_uniad_inference.py` — Full pipeline orchestration script
4. `itri_data_adapter.py` — ITRI format conversion utilities
5. Integration validation scripts — End-to-end testing framework

---

### **Data Flow:**
```
ITRI 4-Camera Images → BEVFormer → BEV Features
ITRI Semantic Map → Lane Queries ↘
ITRI Track Data → Track Queries → MotionFormer → Motion Output
                                              ↓
                              OccFormer → Occupancy Output
                                              ↓
                              Planner → Planning Trajectories
```