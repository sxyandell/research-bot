class GeneticChatbot {
    constructor() {
        this.messageInput = document.getElementById('messageInput');
        this.sendButton = document.getElementById('sendButton');
        this.chatMessages = document.getElementById('chatMessages');
        this.typingIndicator = document.getElementById('typingIndicator');
        
        this.isProcessing = false;
        
        this.initializeEventListeners();
        this.loadStats();
        this.focusInput();
    }
    
    initializeEventListeners() {
        // Send message on button click
        this.sendButton.addEventListener('click', () => this.sendMessage());
        
        // Send message on Enter key
        this.messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        
        // Auto-resize input and handle input changes
        this.messageInput.addEventListener('input', () => {
            this.updateSendButtonState();
        });
        
        // Initial button state
        this.updateSendButtonState();
    }
    
    updateSendButtonState() {
        const hasContent = this.messageInput.value.trim().length > 0;
        this.sendButton.disabled = this.isProcessing || !hasContent;
    }
    
    async loadStats() {
        try {
            const response = await fetch('/stats');
            const stats = await response.json();
            
            // Update stats with animation
            this.animateCounter('totalQTLs', stats.total_qtls || 0);
            this.animateCounter('totalGenes', stats.genes?.length || 0);
            this.animateCounter('totalChromosomes', stats.chromosomes?.length || 0);
            
        } catch (error) {
            console.error('Error loading stats:', error);
            // Set default values if stats fail to load
            document.getElementById('totalQTLs').textContent = '500';
            document.getElementById('totalGenes').textContent = '303';
            document.getElementById('totalChromosomes').textContent = '19';
        }
    }
    
    animateCounter(elementId, targetValue) {
        const element = document.getElementById(elementId);
        const startValue = 0;
        const duration = 2000; // 2 seconds
        const startTime = performance.now();
        
        const animate = (currentTime) => {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            
            // Easing function for smooth animation
            const easeOutQuart = 1 - Math.pow(1 - progress, 4);
            const currentValue = Math.floor(startValue + (targetValue - startValue) * easeOutQuart);
            
            element.textContent = currentValue.toLocaleString();
            
            if (progress < 1) {
                requestAnimationFrame(animate);
            } else {
                element.textContent = targetValue.toLocaleString();
            }
        };
        
        requestAnimationFrame(animate);
    }
    
    async sendMessage() {
        const message = this.messageInput.value.trim();
        if (!message || this.isProcessing) return;
        
        // Clear input and update state
        this.messageInput.value = '';
        this.isProcessing = true;
        this.updateSendButtonState();
        
        // Add user message to chat
        this.addMessage(message, 'user');
        
        // Show typing indicator
        this.showTypingIndicator();
        
        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: message })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            
            // Hide typing indicator
            this.hideTypingIndicator();
            
            // Add bot response
            this.addMessage(data.response, 'bot');
            
        } catch (error) {
            console.error('Error sending message:', error);
            this.hideTypingIndicator();
            this.addMessage('Sorry, I encountered an error while processing your request. Please try again.', 'bot', true);
        } finally {
            this.isProcessing = false;
            this.updateSendButtonState();
            this.focusInput();
        }
    }
    
    addMessage(content, sender, isError = false) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'flex gap-4 animate-slide-in';
        
        if (isError) {
            messageDiv.classList.add('opacity-75');
        }
        
        // Create avatar
        const avatar = document.createElement('div');
        if (sender === 'user') {
            avatar.className = 'flex-shrink-0 w-10 h-10 bg-gradient-to-br from-slate-500 to-slate-700 rounded-full flex items-center justify-center text-xl';
            avatar.textContent = '👤';
        } else {
            avatar.className = 'flex-shrink-0 w-10 h-10 bg-gradient-to-br from-accent-green to-secondary-blue rounded-full flex items-center justify-center text-xl';
            avatar.textContent = '🧬';
        }
        
        // Create content
        const contentDiv = document.createElement('div');
        if (sender === 'user') {
            contentDiv.className = 'flex-1 bg-bg-tertiary/50 rounded-2xl rounded-tl-none p-4 border border-white/10';
        } else {
            contentDiv.className = 'flex-1 bg-bg-secondary/50 rounded-2xl rounded-tl-none p-4 border border-white/10';
        }
        
        const textDiv = document.createElement('div');
        textDiv.className = 'text-slate-300';
        
        // Process content for better formatting
        const processedContent = this.processMessageContent(content);
        textDiv.innerHTML = processedContent;
        
        contentDiv.appendChild(textDiv);
        messageDiv.appendChild(avatar);
        messageDiv.appendChild(contentDiv);
        
        // Add message with animation
        messageDiv.style.opacity = '0';
        messageDiv.style.transform = 'translateY(20px)';
        this.chatMessages.appendChild(messageDiv);
        
        // Animate in
        requestAnimationFrame(() => {
            messageDiv.style.transition = 'all 0.3s ease-out';
            messageDiv.style.opacity = '1';
            messageDiv.style.transform = 'translateY(0)';
        });
        
        // Scroll to bottom
        this.scrollToBottom();
    }
    
    processMessageContent(content) {
        // Convert markdown-style formatting to HTML with Tailwind classes
        let processed = content
            // Bold text **text** or __text__
            .replace(/\*\*(.*?)\*\*/g, '<strong class="text-white font-semibold">$1</strong>')
            .replace(/__(.*?)__/g, '<strong class="text-white font-semibold">$1</strong>')
            // Italic text *text* or _text_
            .replace(/(?<!\*)\*([^*]+)\*(?!\*)/g, '<em class="italic text-slate-200">$1</em>')
            .replace(/(?<!_)_([^_]+)_(?!_)/g, '<em class="italic text-slate-200">$1</em>')
            // Code blocks ```code```
            .replace(/```([\s\S]*?)```/g, '<pre class="bg-bg-primary border border-white/20 rounded-lg p-3 mt-2 mb-2 text-sm font-mono text-accent-green overflow-x-auto"><code>$1</code></pre>')
            // Inline code `code`
            .replace(/`([^`]+)`/g, '<code class="bg-bg-primary px-2 py-1 rounded text-sm font-mono text-accent-green border border-white/20">$1</code>')
            // Line breaks
            .replace(/\n/g, '<br>')
            // Headers
            .replace(/^### (.*$)/gm, '<h4 class="text-lg font-semibold text-accent-purple mt-4 mb-2">$1</h4>')
            .replace(/^## (.*$)/gm, '<h3 class="text-xl font-semibold text-secondary-blue mt-4 mb-2">$1</h3>')
            .replace(/^# (.*$)/gm, '<h2 class="text-2xl font-bold text-accent-green mt-4 mb-2">$1</h2>');
        
        return processed;
    }
    
    showTypingIndicator() {
        this.typingIndicator.classList.remove('hidden');
    }
    
    hideTypingIndicator() {
        this.typingIndicator.classList.add('hidden');
    }
    
    scrollToBottom() {
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }
    
    focusInput() {
        this.messageInput.focus();
    }
}

// Sample query function
function sendSampleQuery(query) {
    const chatbot = window.chatbot;
    if (chatbot) {
        chatbot.messageInput.value = query;
        chatbot.sendMessage();
    }
}

// Initialize chatbot when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.chatbot = new GeneticChatbot();
}); 