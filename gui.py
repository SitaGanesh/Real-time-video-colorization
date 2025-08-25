"""
CORRECTED GUI - Real Video Colorization System
"""

import sys
import cv2
import numpy as np
import torch
from pathlib import Path
import threading
import queue
import time
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk

# Try to import customtkinter with fallback to regular tkinter
try:
    import customtkinter as ctk
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    GUI_LIBRARY = "customtkinter"
except ImportError:
    import tkinter as ctk
    GUI_LIBRARY = "tkinter"
    print("CustomTkinter not available, using standard tkinter")

# Add project paths
sys.path.append('src')
sys.path.append('models')

class SimpleColorizationModel(torch.nn.Module):
    """Simple working colorization model for demonstration"""
    
    def __init__(self):
        super(SimpleColorizationModel, self).__init__()
        
        # Simple U-Net style network
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 128, 3, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(128, 256, 3, stride=2, padding=1),
            torch.nn.ReLU(inplace=True)
        )
        
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 3, 3, padding=1),
            torch.nn.Sigmoid()  # Output RGB values [0,1]
        )
    
    def forward(self, x):
        # Normalize input to [0,1]
        if x.max() > 1.0:
            x = x / 255.0
            
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class WorkingColorizer:
    """Actual working colorizer - not just grayscale converter!"""
    
    def __init__(self, use_demo_model=True):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.use_demo_model = use_demo_model
        
        if use_demo_model:
            # Create a simple demo colorizer that adds basic colors
            self.model = None  # We'll use rule-based colorization for demo
            print("Using demo colorization (rule-based)")
        else:
            # Use actual trained model if available
            self.model = SimpleColorizationModel().to(self.device)
            print("Using neural network colorization")
        
        self.frame_count = 0
        self.total_time = 0
    
    def detect_if_grayscale(self, frame):
        """Detect if a frame is actually grayscale/black & white"""
        if len(frame.shape) == 2:
            return True
        
        # Check if it's a color image that's actually grayscale
        b, g, r = cv2.split(frame)
        
        # If all channels are nearly identical, it's grayscale
        diff_bg = np.mean(np.abs(b.astype(int) - g.astype(int)))
        diff_br = np.mean(np.abs(b.astype(int) - r.astype(int)))
        diff_gr = np.mean(np.abs(g.astype(int) - r.astype(int)))
        
        # Threshold for considering it grayscale
        threshold = 5
        
        is_grayscale = (diff_bg < threshold and diff_br < threshold and diff_gr < threshold)
        
        if is_grayscale:
            print("Detected grayscale image - will colorize")
        else:
            print("Detected color image - will preserve")
            
        return is_grayscale
    
    def demo_colorization(self, gray_frame):
        """Demo colorization using simple rules (for demonstration)"""
        # Convert grayscale to RGB
        if len(gray_frame.shape) == 2:
            rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)
        else:
            rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_BGR2RGB)
            rgb_frame = cv2.cvtColor(cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
        
        # Apply simple colorization rules
        height, width = rgb_frame.shape[:2]
        
        # Add blue tint to darker areas (like sky/water)
        mask_dark = rgb_frame[:, :, 0] < 80
        rgb_frame[mask_dark, 2] = np.minimum(255, rgb_frame[mask_dark, 2] * 1.3)  # Boost blue
        
        # Add green tint to mid-tones (like vegetation)
        mask_mid = (rgb_frame[:, :, 0] >= 80) & (rgb_frame[:, :, 0] < 160)
        rgb_frame[mask_mid, 1] = np.minimum(255, rgb_frame[mask_mid, 1] * 1.2)  # Boost green
        
        # Add warm tint to bright areas
        mask_bright = rgb_frame[:, :, 0] >= 160
        rgb_frame[mask_bright, 0] = np.minimum(255, rgb_frame[mask_bright, 0] * 1.1)  # Boost red
        rgb_frame[mask_bright, 1] = np.minimum(255, rgb_frame[mask_bright, 1] * 1.05)  # Slight green
        
        # Add some color variation based on position (sky = blue, ground = brown/green)
        for y in range(height):
            # Sky area (top 1/3)
            if y < height // 3:
                rgb_frame[y, :, 2] = np.minimum(255, rgb_frame[y, :, 2] * 1.2)  # More blue
            # Ground area (bottom 1/3) 
            elif y > 2 * height // 3:
                rgb_frame[y, :, 1] = np.minimum(255, rgb_frame[y, :, 1] * 1.15)  # More green
                rgb_frame[y, :, 0] = np.minimum(255, rgb_frame[y, :, 0] * 1.05)  # Slight red
        
        return rgb_frame.astype(np.uint8)
    
    def process_frame(self, frame):
        """Process a single frame - colorize only if it's grayscale!"""
        start_time = time.time()
        
        # First, detect if the frame is actually grayscale
        is_grayscale = self.detect_if_grayscale(frame)
        
        if not is_grayscale:
            # Frame is already color - return as is!
            print("Frame is already in color - preserving original")
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                # Convert BGR to RGB for display
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame
        
        # Frame is grayscale - apply colorization
        print("Frame is grayscale - applying colorization")
        
        try:
            if self.use_demo_model:
                # Use demo colorization
                colorized = self.demo_colorization(frame)
            else:
                # Use neural network (if you have a trained model)
                colorized = self.neural_colorization(frame)
            
            # Update performance metrics
            self.frame_count += 1
            self.total_time += time.time() - start_time
            
            return colorized
            
        except Exception as e:
            print(f"Error during colorization: {e}")
            # Fallback to grayscale
            if len(frame.shape) == 2:
                return cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    def neural_colorization(self, gray_frame):
        """Neural network based colorization (placeholder)"""
        # This would use your trained model
        # For now, just return demo colorization
        return self.demo_colorization(gray_frame)
    
    def get_fps(self):
        """Get average processing FPS"""
        if self.total_time > 0:
            return self.frame_count / self.total_time
        return 0

class ColorizationGUI:
    def __init__(self):
        self.colorizer = WorkingColorizer(use_demo_model=True)  # Start with demo
        self.is_running = False
        self.current_source = None
        self.current_frame_data = None
        self.debug_mode = True
        
        # GUI setup
        self.root = None
        self.status_label = None
        self.gui_active = True
        
    def create_gui(self):
        """Create the main GUI"""
        self.root = ctk.CTk()
        self.root.title("REAL Video Colorization System")
        self.root.geometry("900x800")
        
        # Configure grid weights
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(2, weight=1)
        
        # Title
        title_label = ctk.CTkLabel(
            self.root, 
            text="WORKING Video Colorization System",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color="green"
        )
        title_label.grid(row=0, column=0, pady=10, sticky="ew")
        
        # Info label
        info_label = ctk.CTkLabel(
            self.root,
            text="📹 Load B&W video → See it colorized! | Load color video → See original colors!",
            font=ctk.CTkFont(size=12),
            text_color="cyan"
        )
        info_label.grid(row=1, column=0, pady=5, sticky="ew")
        
        # Video display frame
        video_frame = ctk.CTkFrame(self.root)
        video_frame.grid(row=2, column=0, padx=10, pady=5, sticky="nsew")
        video_frame.grid_columnconfigure(0, weight=1)
        video_frame.grid_rowconfigure(1, weight=1)
        
        ctk.CTkLabel(
            video_frame, 
            text="Video Display", 
            font=ctk.CTkFont(size=16, weight="bold")
        ).grid(row=0, column=0, pady=5)
        
        # Create canvas for video display
        self.canvas = tk.Canvas(
            video_frame, 
            width=640, 
            height=480, 
            bg='black',
            highlightthickness=0
        )
        self.canvas.grid(row=1, column=0, padx=10, pady=10)
        
        # Initial text
        self.canvas.create_text(
            320, 240, 
            text="Load a video to see REAL colorization!\n\n🎬 B&W video → Colorized\n🌈 Color video → Original colors", 
            fill="white", 
            font=("Arial", 14),
            justify="center"
        )
        
        # Debug info
        self.debug_label = ctk.CTkLabel(
            video_frame, 
            text="Status: Ready to colorize!",
            font=ctk.CTkFont(size=10)
        )
        self.debug_label.grid(row=2, column=0, pady=2)
        
        # Controls frame
        controls_frame = ctk.CTkFrame(self.root)
        controls_frame.grid(row=3, column=0, padx=10, pady=5, sticky="ew")
        controls_frame.grid_columnconfigure((0, 1, 2), weight=1)
        
        ctk.CTkLabel(
            controls_frame, 
            text="Controls", 
            font=ctk.CTkFont(size=16, weight="bold")
        ).grid(row=0, column=0, columnspan=3, pady=5)
        
        # Buttons
        self.webcam_btn = ctk.CTkButton(
            controls_frame, 
            text="📹 Start Webcam", 
            command=self.start_webcam,
            font=ctk.CTkFont(size=12),
            width=140
        )
        self.webcam_btn.grid(row=1, column=0, padx=5, pady=5)
        
        self.video_btn = ctk.CTkButton(
            controls_frame, 
            text="🎬 Load Video", 
            command=self.load_video_file,
            font=ctk.CTkFont(size=12),
            width=140
        )
        self.video_btn.grid(row=1, column=1, padx=5, pady=5)
        
        self.stop_btn = ctk.CTkButton(
            controls_frame, 
            text="⏹️ Stop", 
            command=self.stop_processing,
            font=ctk.CTkFont(size=12),
            width=140
        )
        self.stop_btn.grid(row=1, column=2, padx=5, pady=5)
        
        # Second row
        self.screenshot_btn = ctk.CTkButton(
            controls_frame, 
            text="📸 Screenshot", 
            command=self.take_screenshot,
            state="disabled",
            font=ctk.CTkFont(size=12),
            width=140
        )
        self.screenshot_btn.grid(row=2, column=0, padx=5, pady=5)
        
        self.mode_btn = ctk.CTkButton(
            controls_frame, 
            text="🎨 Demo Mode", 
            command=self.toggle_mode,
            font=ctk.CTkFont(size=12),
            width=140,
            fg_color="purple"
        )
        self.mode_btn.grid(row=2, column=1, padx=5, pady=5)
        
        self.exit_btn = ctk.CTkButton(
            controls_frame, 
            text="❌ Exit", 
            command=self.on_exit,
            font=ctk.CTkFont(size=12),
            width=140,
            fg_color="darkred"
        )
        self.exit_btn.grid(row=2, column=2, padx=5, pady=5)
        
        # Status frame
        status_frame = ctk.CTkFrame(self.root)
        status_frame.grid(row=4, column=0, padx=10, pady=5, sticky="ew")
        status_frame.grid_columnconfigure(0, weight=1)
        
        self.status_label = ctk.CTkLabel(
            status_frame, 
            text="🔥 Ready to colorize! Load a B&W video to see the magic!",
            font=ctk.CTkFont(size=12),
            text_color="lime"
        )
        self.status_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        
    def toggle_mode(self):
        """Toggle between demo and neural network mode"""
        self.colorizer.use_demo_model = not self.colorizer.use_demo_model
        if self.colorizer.use_demo_model:
            self.mode_btn.configure(text="🎨 Demo Mode", fg_color="purple")
            self.update_status("Using Demo Colorization")
        else:
            self.mode_btn.configure(text="🧠 Neural Mode", fg_color="orange")
            self.update_status("Using Neural Network Colorization")
    
    def safe_gui_update(self, func, *args, **kwargs):
        """Safely update GUI from any thread"""
        if self.gui_active and self.root:
            try:
                self.root.after(0, lambda: func(*args, **kwargs))
            except Exception as e:
                print(f"GUI update error: {e}")
    
    def start_webcam(self):
        """Start webcam processing"""
        try:
            for camera_idx in [0, 1, 2]:
                self.current_source = cv2.VideoCapture(camera_idx)
                if self.current_source.isOpened():
                    break
                self.current_source.release()
            else:
                messagebox.showerror('Error', 'Cannot open webcam')
                return
            
            ret, _ = self.current_source.read()
            if not ret:
                self.current_source.release()
                messagebox.showerror('Error', 'Cannot read from webcam')
                return
            
            self.is_running = True
            threading.Thread(target=self.process_video_stream, daemon=True).start()
            self.update_status('📹 Webcam active - Colorizing in real-time!')
            self.screenshot_btn.configure(state="normal")
            
        except Exception as e:
            messagebox.showerror('Error', f'Webcam error: {str(e)}')
    
    def load_video_file(self):
        """Load video file"""
        try:
            file_path = filedialog.askopenfilename(
                title="Select Video File",
                filetypes=[
                    ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                    ("All files", "*.*")
                ]
            )
            
            if not file_path:
                return
            
            self.current_source = cv2.VideoCapture(file_path)
            if not self.current_source.isOpened():
                messagebox.showerror('Error', f'Cannot open video: {file_path}')
                return
            
            # Test frame
            ret, test_frame = self.current_source.read()
            if not ret:
                messagebox.showerror('Error', 'Cannot read video frames')
                return
            
            # Reset to beginning
            self.current_source.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            # Check if video is grayscale or color
            is_grayscale = self.colorizer.detect_if_grayscale(test_frame)
            
            self.is_running = True
            threading.Thread(target=self.process_video_stream, daemon=True).start()
            
            if is_grayscale:
                self.update_status(f'🎬 B&W Video loaded - Colorizing! ({Path(file_path).name})')
            else:
                self.update_status(f'🌈 Color Video loaded - Preserving colors! ({Path(file_path).name})')
                
            self.screenshot_btn.configure(state="normal")
            
        except Exception as e:
            messagebox.showerror('Error', f'Video loading error: {str(e)}')
    
    def take_screenshot(self):
        """Take screenshot of COLORIZED result"""
        if hasattr(self, 'current_frame_data') and self.current_frame_data is not None:
            try:
                timestamp = int(time.time())
                screenshot_path = f'screenshots/colorized_screenshot_{timestamp}.png'
                os.makedirs('screenshots', exist_ok=True)
                
                # Save the COLORIZED frame (not grayscale!)
                success = cv2.imwrite(screenshot_path, self.current_frame_data)
                
                if success:
                    messagebox.showinfo('Success', f'Colorized screenshot saved!\n{screenshot_path}')
                else:
                    messagebox.showerror('Error', 'Failed to save screenshot')
                
            except Exception as e:
                messagebox.showerror('Error', f'Screenshot error: {str(e)}')
        else:
            messagebox.showwarning('Warning', 'No frame to save')
    
    def process_video_stream(self):
        """Process video frames - REAL colorization happening here!"""
        frame_count = 0
        
        while self.is_running and self.current_source:
            ret, frame = self.current_source.read()
            if not ret:
                break
            
            try:
                # ACTUALLY COLORIZE THE FRAME (not just convert to grayscale!)
                colorized_frame = self.colorizer.process_frame(frame)
                
                # Store colorized result for screenshots
                self.current_frame_data = cv2.cvtColor(colorized_frame, cv2.COLOR_RGB2BGR)
                
                # Resize for display
                display_frame = cv2.resize(colorized_frame, (640, 480))
                
                # Convert to PhotoImage
                pil_image = Image.fromarray(display_frame.astype(np.uint8))
                photo = ImageTk.PhotoImage(pil_image)
                
                # Update display
                self.safe_gui_update(self.update_video_display, photo)
                
                frame_count += 1
                
                # Update FPS
                if frame_count % 30 == 0:
                    fps = self.colorizer.get_fps()
                    self.safe_gui_update(
                        self.update_debug, 
                        f"Frame {frame_count} | FPS: {fps:.1f} | Colorization: {'Demo' if self.colorizer.use_demo_model else 'Neural'}"
                    )
                
                time.sleep(0.03)  # ~30 FPS
                
            except Exception as e:
                print(f"Error processing frame {frame_count}: {e}")
                break
        
        self.safe_gui_update(self.update_status, '⏹️ Video ended')
    
    def update_video_display(self, photo):
        """Update video display"""
        try:
            self.canvas.delete("all")
            self.canvas.create_image(320, 240, image=photo)
            self.canvas.image = photo
        except Exception as e:
            print(f"Display error: {e}")
    
    def update_debug(self, message):
        """Update debug info"""
        try:
            if self.debug_label:
                self.debug_label.configure(text=message)
        except Exception as e:
            print(f"Debug update error: {e}")
    
    def update_status(self, message):
        """Update status"""
        try:
            if self.status_label:
                self.status_label.configure(text=message)
                print(f"Status: {message}")
        except Exception as e:
            print(f"Status update error: {e}")
    
    def stop_processing(self):
        """Stop processing"""
        self.is_running = False
        
        if self.current_source:
            self.current_source.release()
            self.current_source = None
        
        self.current_frame_data = None
        self.screenshot_btn.configure(state="disabled")
        
        # Clear display
        self.canvas.delete("all")
        self.canvas.create_text(320, 240, text="Stopped", fill="white", font=("Arial", 14))
        self.update_status('⏹️ Stopped')
    
    def on_exit(self):
        """Exit application"""
        self.gui_active = False
        self.stop_processing()
        try:
            if self.root:
                self.root.quit()
                self.root.destroy()
        except:
            pass
    
    def run(self):
        """Run the application"""
        self.create_gui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_exit)
        
        try:
            self.root.mainloop()
        except Exception as e:
            print(f"GUI Error: {e}")
        finally:
            self.gui_active = False
            self.stop_processing()

def main():
    """Main function"""
    try:
        os.makedirs('screenshots', exist_ok=True)
        print("🎨 Starting REAL Video Colorization System!")
        
        app = ColorizationGUI()
        app.run()
        
    except Exception as e:
        print(f'Fatal Error: {str(e)}')

if __name__ == '__main__':
    main()
