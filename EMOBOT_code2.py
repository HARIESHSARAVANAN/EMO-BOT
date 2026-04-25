import cv2
import numpy as np
from deepface import DeepFace
import tkinter as tk
from tkinter import scrolledtext
from PIL import Image, ImageTk
import threading
import queue
import requests
import pyttsx3
import time

class EmotionChatbot:
    def __init__(self):
        # Initialize main window
        self.root = tk.Tk()
        self.root.title("EMMA - Emotional Mental Management Assistant")
        self.root.geometry("1400x800")
        self.root.configure(bg='#0f1419')
        
        # State variables
        self.current_emotion = "neutral"
        self.chat_history = []
        self.is_running = True
        self.camera_active = True
        
        # Queues for thread communication
        self.response_queue = queue.Queue()
        
        # Initialize TTS
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)
        
        # Setup UI
        self.setup_ui()
        
        # Start camera thread
        self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
        self.camera_thread.start()
        
        # Start response processor
        self.process_responses()
        
    def setup_ui(self):
        # Professional Header
        header_frame = tk.Frame(self.root, bg='#0a0d11', height=110)
        header_frame.pack(fill=tk.X, side=tk.TOP)
        header_frame.pack_propagate(False)
        
        # College Name - Left Side
        college_frame = tk.Frame(header_frame, bg='#0a0d11')
        college_frame.pack(side=tk.LEFT, padx=30, pady=15)
        
        college_title = tk.Label(
            college_frame, 
            text="SRM VALLIAMMAI ENGINEERING COLLEGE",
            bg='#0a0d11', fg='#64b4f6', 
            font=('Arial', 16, 'bold')
        )
        college_title.pack(anchor='w')
        
        college_subtitle = tk.Label(
            college_frame,
            text="(A Member of SRM Group of Institutions)",
            bg='#0a0d11', fg='#90a4ae',
            font=('Arial', 9)
        )
        college_subtitle.pack(anchor='w', pady=(2, 0))
        
        dept_label = tk.Label(
            college_frame,
            text="Department of Artificial Intelligence and Data Science",
            bg='#0a0d11', fg='#b0bec5',
            font=('Arial', 10)
        )
        dept_label.pack(anchor='w', pady=(5, 0))
        
        # EMMA Badge - Right Side
        badge_frame = tk.Frame(header_frame, bg='#1e3a5f', relief=tk.RAISED, borderwidth=2)
        badge_frame.pack(side=tk.RIGHT, padx=30, pady=20)
        
        emma_label = tk.Label(
            badge_frame,
            text="EMMA - Emotional Assistance",
            bg='#1e3a5f', fg='#ffffff',
            font=('Arial', 13, 'bold'),
            padx=20, pady=12
        )
        emma_label.pack()
        
        status_frame = tk.Frame(badge_frame, bg='#1e3a5f')
        status_frame.pack(pady=(0, 8))
        
        status_dot = tk.Label(status_frame, text="●", bg='#1e3a5f', fg='#4caf50', font=('Arial', 12))
        status_dot.pack(side=tk.LEFT, padx=(15, 5))
        
        status_text = tk.Label(
            status_frame,
            text="SYSTEM ACTIVE",
            bg='#1e3a5f', fg='#4caf50',
            font=('Arial', 9, 'bold')
        )
        status_text.pack(side=tk.LEFT, padx=(0, 15))
        
        # Main container
        main_frame = tk.Frame(self.root, bg='#0f1419')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(10, 20))
        
        # Left side - Camera (65%)
        left_frame = tk.Frame(main_frame, bg='#1c2128', relief=tk.RAISED, borderwidth=2)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        camera_header = tk.Frame(left_frame, bg='#0a0d11', height=45)
        camera_header.pack(fill=tk.X)
        camera_header.pack_propagate(False)
        
        camera_label = tk.Label(
            camera_header, 
            text="📹 LIVE CAMERA FEED", 
            bg='#0a0d11', fg='#64b4f6', 
            font=('Arial', 12, 'bold')
        )
        camera_label.pack(pady=12, padx=15, anchor='w')
        
        self.video_label = tk.Label(left_frame, bg='#0f1419')
        self.video_label.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        self.emotion_label = tk.Label(
            left_frame, 
            text="Emotion: Detecting...", 
            bg='#1c2128', fg='#4caf50', 
            font=('Arial', 11, 'bold')
        )
        self.emotion_label.pack(pady=12)
        
        # Right side - Chat (35%)
        right_frame = tk.Frame(main_frame, bg='#1c2128', relief=tk.RAISED, borderwidth=2)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        chat_header = tk.Frame(right_frame, bg='#0a0d11', height=45)
        chat_header.pack(fill=tk.X)
        chat_header.pack_propagate(False)
        
        chat_label = tk.Label(
            chat_header, 
            text="💬 CONVERSATION", 
            bg='#0a0d11', fg='#64b4f6', 
            font=('Arial', 12, 'bold')
        )
        chat_label.pack(pady=12, padx=15, anchor='w')
        
        # Chat display with custom styling
        self.chat_display = scrolledtext.ScrolledText(
            right_frame, 
            wrap=tk.WORD, 
            bg='#0f1419', 
            fg='#e1e4e8',
            font=('Arial', 10), 
            relief=tk.FLAT, 
            padx=12, 
            pady=12,
            insertbackground='#64b4f6'
        )
        self.chat_display.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.chat_display.config(state=tk.DISABLED)
        
        # Configure chat tags for better styling
        self.chat_display.tag_config('ai_tutor', foreground='#ffa726', font=('Arial', 10, 'bold'))
        self.chat_display.tag_config('student', foreground='#64b4f6', font=('Arial', 10, 'bold'))
        self.chat_display.tag_config('timestamp', foreground='#6c757d', font=('Arial', 8))
        
        # Input area
        input_frame = tk.Frame(right_frame, bg='#1c2128')
        input_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        self.message_entry = tk.Entry(
            input_frame, 
            bg='#2d333b', 
            fg='#e1e4e8', 
            font=('Arial', 10), 
            relief=tk.FLAT,
            insertbackground='#64b4f6',
            borderwidth=2
        )
        self.message_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10), ipady=10)
        self.message_entry.bind('<Return>', lambda e: self.send_message())
        
        send_button = tk.Button(
            input_frame, 
            text="Send ▶", 
            command=self.send_message,
            bg='#1e3a5f', 
            fg='white', 
            font=('Arial', 10, 'bold'),
            relief=tk.FLAT, 
            cursor='hand2', 
            padx=25,
            pady=8,
            borderwidth=0
        )
        send_button.pack(side=tk.RIGHT)
        
        # Hover effect for send button
        def on_enter(e):
            send_button['bg'] = '#2d5a8f'
        
        def on_leave(e):
            send_button['bg'] = '#1e3a5f'
        
        send_button.bind('<Enter>', on_enter)
        send_button.bind('<Leave>', on_leave)
        
        # Bottom status bar
        self.status_label = tk.Label(
            self.root, 
            text="🟢 Ready  |  ESC: Exit  |  ENTER: Send  |  BACKSPACE: Delete", 
            bg='#0a0d11', 
            fg='#4caf50', 
            font=('Arial', 9)
        )
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
        
    def camera_loop(self):
        """Camera capture and emotion detection loop"""
        cap = cv2.VideoCapture(0)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        frame_count = 0
        
        # Emotion colors for overlay
        emotion_colors = {
            'happy': (100, 220, 150),
            'sad': (255, 150, 120),
            'angry': (100, 120, 255),
            'surprise': (255, 200, 100),
            'fear': (200, 120, 255),
            'disgust': (150, 200, 100),
            'neutral': (160, 180, 200)
        }
        
        while self.is_running and self.camera_active:
            ret, frame = cap.read()
            if not ret:
                continue
                
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            # Detect emotion every 30 frames
            if frame_count % 30 == 0 and len(faces) > 0:
                try:
                    result = DeepFace.analyze(
                        frame, 
                        actions=['emotion'], 
                        enforce_detection=False
                    )
                    if isinstance(result, list):
                        result = result[0]
                    self.current_emotion = result['dominant_emotion']
                    
                    # Update emotion label with emoji
                    emotion_emojis = {
                        'happy': '😊', 'sad': '😔', 'angry': '😠',
                        'surprise': '😲', 'fear': '😰', 'disgust': '😖',
                        'neutral': '😐'
                    }
                    emoji = emotion_emojis.get(self.current_emotion, '😐')
                    
                    self.root.after(0, lambda: self.emotion_label.config(
                        text=f"Emotion: {self.current_emotion.upper()} {emoji}"
                    ))
                except Exception as e:
                    print(f"Emotion detection error: {e}")
            
            # Draw professional rectangles around faces
            for (x, y, w, h) in faces:
                color = emotion_colors.get(self.current_emotion, (160, 180, 200))
                
                # Main rectangle
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                
                # Corner accents
                corner_length = 25
                cv2.line(frame, (x, y), (x+corner_length, y), color, 5)
                cv2.line(frame, (x, y), (x, y+corner_length), color, 5)
                cv2.line(frame, (x+w, y), (x+w-corner_length, y), color, 5)
                cv2.line(frame, (x+w, y), (x+w, y+corner_length), color, 5)
                cv2.line(frame, (x, y+h), (x+corner_length, y+h), color, 5)
                cv2.line(frame, (x, y+h), (x, y+h-corner_length), color, 5)
                cv2.line(frame, (x+w, y+h), (x+w-corner_length, y+h), color, 5)
                cv2.line(frame, (x+w, y+h), (x+w, y+h-corner_length), color, 5)
                
                # Emotion label on top
                label = f"{self.current_emotion.upper()}"
                (text_width, text_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                )
                
                # Background for text
                cv2.rectangle(
                    frame, 
                    (x, y-35), 
                    (x+text_width+20, y), 
                    color, 
                    -1
                )
                cv2.putText(
                    frame, 
                    label, 
                    (x+10, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (255, 255, 255), 
                    2
                )
            
            # Convert and display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = img.resize((640, 480))
            imgtk = ImageTk.PhotoImage(image=img)
            
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
            
            frame_count += 1
            time.sleep(0.03)
        
        cap.release()
        
    def add_message_to_chat(self, sender, message, tag):
        """Add message to chat display with better formatting"""
        self.chat_display.config(state=tk.NORMAL)
        
        timestamp = time.strftime("%H:%M:%S")
        
        # Add spacing between messages
        if self.chat_display.get('1.0', tk.END).strip():
            self.chat_display.insert(tk.END, "\n")
        
        # Add timestamp
        self.chat_display.insert(tk.END, f"[{timestamp}] ", 'timestamp')
        
        # Add sender with badge-like appearance
        sender_display = f"{sender}: "
        self.chat_display.insert(tk.END, sender_display, tag)
        
        # Add message
        self.chat_display.insert(tk.END, f"{message}\n", 'message')
        
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)
        
    def send_message(self):
        """Send user message"""
        message = self.message_entry.get().strip()
        if not message:
            return
            
        self.message_entry.delete(0, tk.END)
        
        # Display user message
        self.add_message_to_chat("STUDENT", message, 'student')
        self.status_label.config(text="🔄 AI is thinking...", fg='#ffa726')
        
        # Send to AI in background
        threading.Thread(
            target=self.get_ai_response, 
            args=(message,), 
            daemon=True
        ).start()
        
    def get_ai_response(self, user_message):
        """Get response from Ollama"""
        try:
            # Prepare system prompt with emotion context
            system_prompt = f"""You are EMMA (Emotional Mental Management Assistant), a warm and supportive AI companion.

Current user emotion: {self.current_emotion}

Respond in 2-3 sentences with empathy and practical advice. Be direct, warm, and helpful."""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
            
            # Add recent chat history
            for msg in self.chat_history[-4:]:
                messages.append(msg)
            
            messages.append({"role": "user", "content": user_message})
            
            # Call Ollama API
            response = requests.post(
                'http://localhost:11434/api/chat',
                json={
                    "model": "llama3.2:3b",
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": 0.8,
                        "num_predict": 150
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                ai_response = response.json()['message']['content']
                
                # Update chat history
                self.chat_history.append({"role": "user", "content": user_message})
                self.chat_history.append({"role": "assistant", "content": ai_response})
                
                # Queue response for display
                self.response_queue.put(ai_response)
            else:
                error_msg = f"Error: {response.status_code}"
                self.response_queue.put(error_msg)
                
        except requests.exceptions.ConnectionError:
            error_msg = "⚠ Cannot connect to Ollama. Please ensure Ollama is running."
            self.response_queue.put(error_msg)
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            self.response_queue.put(error_msg)
    
    def process_responses(self):
        """Process queued responses"""
        try:
            while True:
                response = self.response_queue.get_nowait()
                
                # Display in chat
                self.add_message_to_chat("AI TUTOR", response, 'ai_tutor')
                
                # Speak response (only first sentence)
                if not response.startswith('⚠'):
                    first_sentence = response.split('.')[0].strip() + '.'
                    if len(first_sentence) > 5:
                        threading.Thread(
                            target=self.speak, 
                            args=(first_sentence,), 
                            daemon=True
                        ).start()
                
                self.status_label.config(text="🟢 Ready", fg='#4caf50')
        except queue.Empty:
            pass
        
        # Schedule next check
        if self.is_running:
            self.root.after(100, self.process_responses)
    
    def speak(self, text):
        """Text-to-speech"""
        try:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception as e:
            print(f"TTS error: {e}")
    
    def run(self):
        """Start the application"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Add welcome message
        welcome = """Welcome! I'm EMMA, your emotional support companion. I'm here to listen and help. What's on your mind today?"""
        self.add_message_to_chat("AI TUTOR", welcome, 'ai_tutor')
        
        print("\n" + "="*70)
        print("🚀 EMMA - EMOTIONAL MENTAL MANAGEMENT ASSISTANT")
        print("="*70)
        print("✓ Camera active")
        print("✓ AI chat engine ready")
        print("✓ TTS engine initialized")
        print("="*70 + "\n")
        
        self.root.mainloop()
    
    def on_closing(self):
        """Clean shutdown"""
        print("\n" + "="*70)
        print("SHUTTING DOWN SYSTEM")
        print("="*70)
        
        self.is_running = False
        self.camera_active = False
        
        try:
            self.tts_engine.stop()
        except:
            pass
        
        print("✓ Shutdown complete")
        print("="*70 + "\n")
        
        self.root.destroy()

if __name__ == "__main__":
    app = EmotionChatbot()
    app.run()