import cv2
import os

def save_snapshot(frame, frame_idx, video_name, alert=False, output_dir="output/snapshots"):
    """Saves a snapshot of the current frame."""
    os.makedirs(output_dir, exist_ok=True)
    prefix = "ALERT_" if alert else "DEBUG_"
    path = os.path.join(output_dir, f"{prefix}{video_name}_frame{frame_idx}.jpg")
    cv2.imwrite(path, frame)
    print(f"[ALERT] Snapshot saved: {path}")

def draw_annotations(frame, tracked_persons, alert, is_snatch, snatch_prob, current_ids, unique_ids, thief_id=-1):
    """Draws boxes, IDs, and alert status on the frame."""
    # Global flag overlay
    alert_text = "ALERT: SNATCHING DETECTED" if is_snatch else "Normal"
    color = (0, 0, 255) if is_snatch else (0, 255, 0)
    cv2.putText(frame, f"System: {alert_text} ({snatch_prob*100:.1f}%)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    for p in tracked_persons:
        x1, y1, x2, y2 = p['box']
        tid = p['id']
        current_ids.append(tid)
        unique_ids.add(tid)
        
        # Color the fastest moving person (thief) red, and the bystanders green
        if is_snatch and tid == thief_id:
            bcolor = (0, 0, 255) # Red for Thief
            label_text = f"THIEF ID: {tid}"
        else:
            bcolor = (0, 255, 0) # Green for Bystanders
            label_text = f"ID: {tid}"
            
        cv2.rectangle(frame, (x1, y1), (x2, y2), bcolor, 2)
        cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bcolor, 2)
    
    return frame
