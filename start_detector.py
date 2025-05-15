import os
import sys
import warnings
import logging
import traceback

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.ERROR)
logging.getLogger("ultralytics").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

def main():
    try:
        from vehicle_speed_detector import VehicleSpeedDetector, main as detector_main
        
        detector_main()
        
    except Exception as e:
        print(f"runtime error: {e}")
        traceback.print_exc()
    
    print("\nPress any key to exit.")
    input()

if __name__ == "__main__":
    main() 
