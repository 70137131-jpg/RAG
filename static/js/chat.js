// RAG Chatbot - Chat Interface JavaScript

class ChatBot {
    constructor() {
        this.chatMessages = document.getElementById('chatMessages');
        this.chatForm = document.getElementById('chatForm');
        this.questionInput = document.getElementById('questionInput');
        this.sendBtn = document.getElementById('sendBtn');
        this.clearBtn = document.getElementById('clearBtn');
        this.statsBtn = document.getElementById('statsBtn');
        this.typingIndicator = document.getElementById('typingIndicator');
        this.charCount = document.getElementById('charCount');
        this.topKSelect = document.getElementById('topK');
        this.statsModal = document.getElementById('statsModal');
        this.chatContainer = document.querySelector('.chat-container');

        this.isLoading = false;
        this.initEventListeners();
        this.autoResizeTextarea();
        this.loadChatHistory();
    }

    initEventListeners() {
        // Form submission
        if (this.chatForm) {
            this.chatForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.sendMessage();
            });
        }

        // Input validation and auto-resize
        if (this.questionInput) {
            this.questionInput.addEventListener('input', () => {
                const value = this.questionInput.value.trim();
                this.sendBtn.disabled = value.length === 0 || this.isLoading;
                if (this.charCount) {
                    this.charCount.textContent = this.questionInput.value.length;
                }
                this.autoResizeTextarea();
            });

            // Enter key to send (Shift+Enter for new line)
            this.questionInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    if (!this.sendBtn.disabled && !this.isLoading) {
                        this.sendMessage();
                    }
                }
            });
        }

        // Clear chat
        if (this.clearBtn) {
            this.clearBtn.addEventListener('click', () => {
                if (confirm('Are you sure you want to clear the chat history?')) {
                    this.clearChat();
                }
            });
        }

        // Stats modal
        if (this.statsBtn) {
            this.statsBtn.addEventListener('click', () => {
                this.showStats();
            });
        }

        // Close modal
        if (this.statsModal) {
            const modalClose = this.statsModal.querySelector('.modal-close');
            if (modalClose) {
                modalClose.addEventListener('click', () => {
                    this.statsModal.classList.remove('active');
                });
            }

            // Close modal on outside click
            this.statsModal.addEventListener('click', (e) => {
                if (e.target === this.statsModal) {
                    this.statsModal.classList.remove('active');
                }
            });
        }

        // Example question buttons
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('example-btn')) {
                const question = e.target.dataset.question;
                if (question && this.questionInput) {
                    this.questionInput.value = question;
                    this.sendBtn.disabled = false;
                    if (this.charCount) {
                        this.charCount.textContent = question.length;
                    }
                    this.questionInput.focus();
                    this.autoResizeTextarea();
                }
            }
        });

        // Escape key to close modal
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.statsModal && this.statsModal.classList.contains('active')) {
                this.statsModal.classList.remove('active');
            }
        });
    }

    autoResizeTextarea() {
        if (this.questionInput) {
            this.questionInput.style.height = 'auto';
            this.questionInput.style.height = Math.min(this.questionInput.scrollHeight, 120) + 'px';
        }
    }

    async loadChatHistory() {
        try {
            const response = await fetch('/api/history');
            const data = await response.json();

            if (data.success && data.history && data.history.length > 0) {
                data.history.forEach(item => {
                    this.addMessage(item.question, 'user', null, false);
                    this.addMessage(item.answer, 'bot', null, false);
                });
            }
        } catch (error) {
            console.error('Error loading chat history:', error);
        }
    }

    async sendMessage() {
        const question = this.questionInput.value.trim();
        if (!question || this.isLoading) return;

        const topK = this.topKSelect ? parseInt(this.topKSelect.value) : 3;

        // Set loading state
        this.isLoading = true;
        this.sendBtn.disabled = true;

        // Add user message to chat
        this.addMessage(question, 'user');

        // Clear input
        this.questionInput.value = '';
        if (this.charCount) {
            this.charCount.textContent = '0';
        }
        this.autoResizeTextarea();

        // Show typing indicator
        this.showTyping();

        try {
            // Send request to API
            const response = await fetch('/api/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    question: question,
                    top_k: topK
                })
            });

            const data = await response.json();

            // Hide typing indicator
            this.hideTyping();

            if (data.success) {
                // Add bot response with sources
                this.addMessage(data.answer, 'bot', data.sources);
            } else {
                // Show error
                this.addMessage(
                    `Sorry, I encountered an error: ${data.error || 'Unknown error'}`,
                    'bot'
                );
            }
        } catch (error) {
            this.hideTyping();
            this.addMessage(
                'Sorry, there was a network error. Please check if the server is running and try again.',
                'bot'
            );
            console.error('Error:', error);
        } finally {
            // Reset loading state
            this.isLoading = false;
            this.sendBtn.disabled = this.questionInput.value.trim().length === 0;
        }
    }

    addMessage(text, type, sources = null, animate = true) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}-message`;
        if (animate) {
            messageDiv.style.animation = 'slideIn 0.3s ease';
        }

        // Avatar
        const avatarDiv = document.createElement('div');
        avatarDiv.className = `message-avatar ${type}-avatar`;

        if (type === 'bot') {
            avatarDiv.innerHTML = `
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <rect x="3" y="11" width="18" height="10" rx="2"/>
                    <circle cx="12" cy="5" r="2"/>
                    <path d="M12 7v4"/>
                </svg>
            `;
        } else {
            avatarDiv.innerHTML = `
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
                    <circle cx="12" cy="7" r="4"></circle>
                </svg>
            `;
        }

        // Content
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';

        const textDiv = document.createElement('div');
        textDiv.className = 'message-text';

        // Add text with proper formatting
        const textParagraph = document.createElement('p');
        textParagraph.textContent = text;
        textDiv.appendChild(textParagraph);

        // Add sources if available
        if (sources && sources.length > 0) {
            const sourcesDiv = document.createElement('div');
            sourcesDiv.className = 'sources';

            const sourcesHeader = document.createElement('div');
            sourcesHeader.className = 'sources-header';
            sourcesHeader.innerHTML = `
                <span class="sources-title">Sources (${sources.length})</span>
                <button class="sources-toggle" onclick="this.parentElement.parentElement.classList.toggle('collapsed')">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <polyline points="6 9 12 15 18 9"></polyline>
                    </svg>
                </button>
            `;
            sourcesDiv.appendChild(sourcesHeader);

            const sourcesList = document.createElement('div');
            sourcesList.className = 'sources-list';

            sources.forEach((source, index) => {
                const sourceItem = document.createElement('div');
                sourceItem.className = 'source-item';

                const similarityClass = source.similarity >= 80 ? 'high' : source.similarity >= 60 ? 'medium' : 'low';

                sourceItem.innerHTML = `
                    <div class="source-header">
                        <span class="source-id">Source ${index + 1}</span>
                        <span class="source-similarity ${similarityClass}">${source.similarity}% match</span>
                    </div>
                    <div class="source-text">${this.escapeHtml(source.text)}</div>
                `;

                sourcesList.appendChild(sourceItem);
            });

            sourcesDiv.appendChild(sourcesList);
            textDiv.appendChild(sourcesDiv);
        }

        contentDiv.appendChild(textDiv);

        messageDiv.appendChild(avatarDiv);
        messageDiv.appendChild(contentDiv);

        // Insert before typing indicator
        if (this.typingIndicator && this.typingIndicator.parentNode === this.chatMessages) {
            this.chatMessages.insertBefore(messageDiv, this.typingIndicator);
        } else {
            this.chatMessages.appendChild(messageDiv);
        }

        // Scroll to bottom
        this.scrollToBottom();
    }

    showTyping() {
        if (this.typingIndicator) {
            this.typingIndicator.style.display = 'flex';
            this.scrollToBottom();
        }
    }

    hideTyping() {
        if (this.typingIndicator) {
            this.typingIndicator.style.display = 'none';
        }
    }

    scrollToBottom() {
        setTimeout(() => {
            if (this.chatContainer) {
                this.chatContainer.scrollTop = this.chatContainer.scrollHeight;
            }
        }, 50);
    }

    async clearChat() {
        try {
            const response = await fetch('/api/clear', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            const data = await response.json();

            if (data.success) {
                // Remove all messages except the welcome message
                const messages = this.chatMessages.querySelectorAll('.message:not(.typing-indicator)');
                messages.forEach((msg, index) => {
                    if (index > 0) { // Keep first message (welcome)
                        msg.remove();
                    }
                });
            }
        } catch (error) {
            console.error('Error clearing chat:', error);
            alert('Failed to clear chat history');
        }
    }

    async showStats() {
        if (!this.statsModal) return;

        this.statsModal.classList.add('active');
        const statsContent = document.getElementById('statsContent');
        if (statsContent) {
            statsContent.innerHTML = '<div class="loading">Loading statistics...</div>';
        }

        try {
            const response = await fetch('/api/stats');
            const data = await response.json();

            if (data.success && statsContent) {
                const stats = data.stats;
                statsContent.innerHTML = `
                    <div class="stat-item">
                        <span class="stat-label">Documents Indexed</span>
                        <span class="stat-value">${stats.total_documents ? stats.total_documents.toLocaleString() : 'N/A'}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Embedding Model</span>
                        <span class="stat-value">${stats.embedding_model || 'N/A'}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">LLM Model</span>
                        <span class="stat-value">${stats.llm_model || 'N/A'}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Collection Name</span>
                        <span class="stat-value">${stats.collection_name || 'N/A'}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Total Sessions</span>
                        <span class="stat-value">${stats.total_sessions || 0}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Total Queries</span>
                        <span class="stat-value">${stats.total_queries || 0}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Avg Response Time</span>
                        <span class="stat-value">${stats.avg_response_time_ms ? stats.avg_response_time_ms + 'ms' : 'N/A'}</span>
                    </div>
                `;
            } else if (statsContent) {
                statsContent.innerHTML = `<div class="loading">Failed to load statistics: ${data.error || 'Unknown error'}</div>`;
            }
        } catch (error) {
            console.error('Error loading stats:', error);
            if (statsContent) {
                statsContent.innerHTML = '<div class="loading">Error loading statistics. Is the server running?</div>';
            }
        }
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Initialize chatbot when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.chatbot = new ChatBot();
});
