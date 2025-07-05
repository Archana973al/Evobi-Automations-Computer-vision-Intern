# Evobi-Automations-Computer-vision-Intern

## 👤 Person Tracking System

A computer vision system that tracks a specific person in video streams with movement visualization.

## 🚀 Features
- **Smart Detection**  
  ✅ Reference image matching  
  ✅ Color-based (green jacket + white shirt)  
- **Visual Tracking**  
  🟩 Real-time bounding box  
  🟦 Movement flow line  
- **Interactive**  
  🖱️ Manual target selection  
  🔘 Toggle views (box/flow)  
- **Metrics**  
  ⏱️ First appearance timestamp  
  ⚡ Processing FPS counter  

## 🛠️ Installation
```bash
git clone https://github.com/Archana973al/Evobi-Automations-Computer-vision-Intern.git
cd Evobi-Automations-Computer-vision-Intern
pip install -r requirements.txt
```

## Usage
### Basic Tracking
```bash
python main.py -i input.mp4 -o output.mp4 
```
 
## Interactive Mode
Run: python main.py

In first frame:

🖱️ Click-drag to select target

💾 Press s to confirm

During tracking:

b → Toggle bounding box

f → Toggle flow line

q → Quit

## Sample Output:
```bash
https://drive.google.com/file/d/1qjcT4jk3bbnp8ZZayQCPVvbIfM9rSD3x/view?usp=sharing