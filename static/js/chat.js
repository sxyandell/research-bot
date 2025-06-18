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
        this.sendButton.style.opacity = (this.isProcessing || !hasContent) ? '0.5' : '1';
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
        messageDiv.className = `message ${sender}-message`;
        
        if (isError) {
            messageDiv.classList.add('error-message');
        }
        
        // Create avatar
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.textContent = sender === 'user' ? '👤' : '🧬';
        
        // Create content
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        
        const textDiv = document.createElement('div');
        textDiv.className = 'message-text';
        
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
        // Convert markdown-style formatting to HTML
        let processed = content
            // Bold text **text** or __text__
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/__(.*?)__/g, '<strong>$1</strong>')
            // Italic text *text* or _text_
            .replace(/(?<!\*)\*([^*]+)\*(?!\*)/g, '<em>$1</em>')
            .replace(/(?<!_)_([^_]+)_(?!_)/g, '<em>$1</em>')
            // Code blocks ```code```
            .replace(/```([\s\S]*?)```/g, '<code class="code-block">$1</code>')
            // Inline code `code`
            .replace(/`([^`]+)`/g, '<code>$1</code>')
            // Line breaks
            .replace(/\n/g, '<br>')
            // Headers
            .replace(/^### (.*$)/gm, '<h4>$1</h4>')
            .replace(/^## (.*$)/gm, '<h3>$1</h3>')
            .replace(/^# (.*$)/gm, '<h2>$1</h2>');
        
        return processed;
    }
    
    showTypingIndicator() {
        this.typingIndicator.style.display = 'flex';
        this.scrollToBottom();
    }
    
    hideTypingIndicator() {
        this.typingIndicator.style.display = 'none';
    }
    
    scrollToBottom() {
        setTimeout(() => {
            this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
        }, 100);
    }
    
    focusInput() {
        setTimeout(() => {
            this.messageInput.focus();
        }, 100);
    }
}

// Sample query function for sidebar buttons
function sendSampleQuery(query) {
    const chatbot = window.chatbotInstance;
    if (chatbot && !chatbot.isProcessing) {
        chatbot.messageInput.value = query;
        chatbot.sendMessage();
    }
}

// Add some CSS for the processed content
const style = document.createElement('style');
style.textContent = `
    .message-text h2 {
        color: var(--accent-green);
        font-size: 1.2rem;
        margin: 0.5rem 0;
        font-weight: 600;
    }
    
    .message-text h3 {
        color: var(--accent-green);
        font-size: 1.1rem;
        margin: 0.5rem 0;
        font-weight: 600;
    }
    
    .message-text h4 {
        color: var(--secondary-blue);
        font-size: 1rem;
        margin: 0.4rem 0;
        font-weight: 600;
    }
    
    .message-text code {
        background: var(--bg-secondary);
        padding: 0.2rem 0.4rem;
        border-radius: 4px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        color: var(--accent-orange);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .message-text .code-block {
        display: block;
        background: var(--bg-secondary);
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        white-space: pre-wrap;
        overflow-x: auto;
        border: 1px solid rgba(255, 255, 255, 0.1);
        line-height: 1.4;
    }
    
    .message-text em {
        font-style: italic;
        color: var(--text-secondary);
    }
    
    .error-message .message-content {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(239, 68, 68, 0.05));
        border-color: rgba(239, 68, 68, 0.3);
    }
    
    .error-message .message-avatar {
        background: #ef4444;
    }
    
    /* Custom scrollbar for better UX */
    .chat-messages::-webkit-scrollbar {
        width: 8px;
    }
    
    .chat-messages::-webkit-scrollbar-track {
        background: var(--bg-secondary);
        border-radius: 4px;
    }
    
    .chat-messages::-webkit-scrollbar-thumb {
        background: linear-gradient(45deg, var(--accent-green), var(--secondary-blue));
        border-radius: 4px;
    }
    
    .chat-messages::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(45deg, var(--secondary-blue), var(--accent-purple));
    }
    
    /* Loading states */
    .send-button:disabled {
        cursor: not-allowed;
        opacity: 0.5;
    }
    
    /* Enhance query buttons */
    .query-button:active {
        transform: translateX(2px) scale(0.98);
    }
    
    /* Add pulse animation for DNA icon */
    .dna-icon {
        transition: transform 0.3s ease;
    }
    
    .dna-icon:hover {
        transform: scale(1.1) rotate(5deg);
    }
    
    /* Add shimmer effect for loading stats */
    .stat-number.loading {
        background: linear-gradient(90deg, transparent, rgba(16, 185, 129, 0.2), transparent);
        background-size: 200px 100%;
        animation: shimmer 1.5s infinite;
        border-radius: 4px;
    }
    
    @keyframes shimmer {
        0% { background-position: -200px 0; }
        100% { background-position: calc(200px + 100%) 0; }
    }
    
    /* Enhance welcome message */
    .welcome-message .message-content {
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.1);
    }
    
    /* Add typing animation enhancement */
    .typing-indicator {
        animation: fadeInUp 0.3s ease-out;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
`;

document.head.appendChild(style);

// Initialize the chatbot when the page loads
document.addEventListener('DOMContentLoaded', () => {
    window.chatbotInstance = new GeneticChatbot();
    
    // Add some particle interaction
    const particles = document.querySelectorAll('.particle');
    particles.forEach((particle, index) => {
        particle.addEventListener('mouseenter', () => {
            particle.style.transform = 'scale(2)';
            particle.style.opacity = '1';
        });
        
        particle.addEventListener('mouseleave', () => {
            particle.style.transform = 'scale(1)';
            particle.style.opacity = '0.6';
        });
    });
    
    // Add keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        // Focus input with Ctrl/Cmd + K
        if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
            e.preventDefault();
            window.chatbotInstance.focusInput();
        }
        
        // Clear chat with Ctrl/Cmd + Shift + C
        if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'C') {
            e.preventDefault();
            if (confirm('Clear all chat messages?')) {
                const welcomeMessage = document.querySelector('.welcome-message');
                window.chatbotInstance.chatMessages.innerHTML = '';
                if (welcomeMessage) {
                    window.chatbotInstance.chatMessages.appendChild(welcomeMessage.cloneNode(true));
                }
            }
        }
    });
});

// Add some easter eggs for fun
let konamiCode = [];
const konamiSequence = ['ArrowUp', 'ArrowUp', 'ArrowDown', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'ArrowLeft', 'ArrowRight', 'KeyB', 'KeyA'];

document.addEventListener('keydown', (e) => {
    konamiCode.push(e.code);
    if (konamiCode.length > konamiSequence.length) {
        konamiCode = konamiCode.slice(1);
    }
    
    if (JSON.stringify(konamiCode) === JSON.stringify(konamiSequence)) {
        // Easter egg: DNA dance
        const dnaIcon = document.querySelector('.dna-icon');
        if (dnaIcon) {
            dnaIcon.style.animation = 'none';
            setTimeout(() => {
                dnaIcon.style.animation = 'spin 1s linear 3, pulse 2s ease-in-out infinite';
            }, 10);
        }
        
        // Add some fun particles
        for (let i = 0; i < 10; i++) {
            setTimeout(() => {
                const particle = document.createElement('div');
                particle.innerHTML = '🧬';
                particle.style.position = 'fixed';
                particle.style.left = Math.random() * window.innerWidth + 'px';
                particle.style.top = '-50px';
                particle.style.fontSize = '2rem';
                particle.style.zIndex = '1000';
                particle.style.pointerEvents = 'none';
                particle.style.animation = 'float 3s linear forwards';
                document.body.appendChild(particle);
                
                setTimeout(() => {
                    particle.remove();
                }, 3000);
            }, i * 200);
        }
        
        konamiCode = [];
    }
});

// CSS keyframes are handled in the style element above 