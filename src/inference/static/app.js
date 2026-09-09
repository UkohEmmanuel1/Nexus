                        // State Management
let currentTab = 'chat';
let activeKeys = [];

// DOM Elements
document.addEventListener('DOMContentLoaded', () => {
    // Restore saved API Key and Admin Key from LocalStorage
    const savedAdminKey = localStorage.getItem('nexus_admin_key');
    if (savedAdminKey) {
        document.getElementById('admin-key-input').value = savedAdminKey;
    }
    
    // Auto-adjust textarea height
    const textarea = document.getElementById('chat-input');
    textarea.addEventListener('input', () => {
        textarea.style.height = 'auto';
        textarea.style.height = (textarea.scrollHeight - 16) + 'px';
    });

    textarea.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Load active keys list if possible
    fetchKeysList();
});

// Tab Navigation
function switchTab(tabId) {
    document.querySelectorAll('.nav-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.view-section').forEach(view => view.classList.remove('active'));

    document.getElementById(`tab-${tabId}`).classList.add('active');
    document.getElementById(`view-${tabId}`).classList.add('active');
    currentTab = tabId;

    if (tabId === 'keys') {
        fetchKeysList();
    }
}

// Code Integration Tab Navigation
function switchCodeTab(lang) {
    document.querySelectorAll('.code-tab').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.code-box pre').forEach(pre => pre.classList.add('hidden'));

    event.target.classList.add('active');
    document.getElementById(`code-guide-${lang}`).classList.remove('hidden');
}

// Fetch list of keys from API
async function fetchKeysList() {
    const adminKey = document.getElementById('admin-key-input').value.stripOrEmpty();
    
    // Save admin key for convenience
    if (adminKey) {
        localStorage.setItem('nexus_admin_key', adminKey);
    }

    try {
        const headers = { 'Content-Type': 'application/json' };
        if (adminKey) {
            headers['X-API-Key'] = adminKey;
        }

        const res = await fetch('/v1/keys', { headers });
        if (res.ok) {
            activeKeys = await res.json();
            renderKeysTable();
        }
    } catch (err) {
        console.error('Error fetching keys list:', err);
    }
}

// Helper to sanitize/strip inputs
String.prototype.stripOrEmpty = function() {
    return this ? this.trim() : '';
};

// Render keys into active keys table
function renderKeysTable() {
    const tbody = document.getElementById('keys-table-body');
    tbody.innerHTML = '';

    if (activeKeys.length === 0) {
        tbody.innerHTML = `<tr><td colspan="4" class="empty-table-text">No keys found. Generate one or set your Admin Key to load existing keys.</td></tr>`;
        return;
    }

    activeKeys.forEach(item => {
        const tr = document.createElement('tr');
        
        // Mask the key token except for prefix and last 4 characters
        const keyDisplay = item.key.substring(0, 12) + '...' + item.key.substring(item.key.length - 4);
        const createdDate = new Date(item.created_at * 1000).toLocaleString();
        
        const isRevoked = item.status === 'revoked';
        const statusSpan = isRevoked 
            ? `<span style="color: var(--danger-color); font-weight: 500;">Revoked</span>`
            : `<span style="color: var(--success-color); font-weight: 500;">Active</span>`;

        tr.innerHTML = `
            <td><strong>${escapeHtml(item.name)}</strong></td>
            <td style="font-family: var(--font-mono); font-size: 0.8rem;">${keyDisplay}</td>
            <td style="color: var(--text-secondary); font-size: 0.8rem;">${createdDate}</td>
            <td>
                ${isRevoked 
                    ? statusSpan 
                    : `<button class="revoke-btn" onclick="revokeKey('${item.key}')">Revoke</button>`
                }
            </td>
        `;
        tbody.appendChild(tr);
    });
}

// Generate a new API Key
async function generateApiKey() {
    const name = document.getElementById('key-name-input').value.stripOrEmpty();
    const adminKey = document.getElementById('admin-key-input').value.stripOrEmpty();

    if (!name) {
        alert('Please specify a key name.');
        return;
    }

    if (adminKey) {
        localStorage.setItem('nexus_admin_key', adminKey);
    }

    try {
        const headers = { 'Content-Type': 'application/json' };
        if (adminKey) {
            headers['Authorization'] = `Bearer ${adminKey}`;
        }

        const res = await fetch('/v1/keys/generate', {
            method: 'POST',
            headers,
            body: JSON.stringify({ name })
        });

        if (res.ok) {
            const data = await res.json();
            
            // Show new key to user
            const displayDiv = document.getElementById('new-key-display');
            displayDiv.classList.remove('hidden');
            document.getElementById('new-key-value').value = data.key;
            
            // Store locally for test chats
            localStorage.setItem('nexus_active_user_key', data.key);

            // Reset inputs & refresh table
            document.getElementById('key-name-input').value = '';
            fetchKeysList();
        } else {
            const err = await res.json();
            alert(`Failed to generate key: ${err.detail || res.statusText}`);
        }
    } catch (err) {
        alert(`Request error: ${err.message}`);
    }
}

// Revoke an API Key
async function revokeKey(key) {
    if (!confirm('Are you sure you want to revoke this API key? This action is permanent.')) {
        return;
    }

    const adminKey = document.getElementById('admin-key-input').value.stripOrEmpty();
    const userKey = localStorage.getItem('nexus_active_user_key') || adminKey;

    try {
        const headers = { 
            'Content-Type': 'application/json'
        };
        // Use userKey or adminKey to authorize revocation
        if (userKey) {
            headers['X-API-Key'] = userKey;
        }

        const res = await fetch('/v1/keys/revoke', {
            method: 'POST',
            headers,
            body: JSON.stringify({ key })
        });

        if (res.ok) {
            fetchKeysList();
        } else {
            const err = await res.json();
            alert(`Failed to revoke key: ${err.detail || res.statusText}`);
        }
    } catch (err) {
        alert(`Request error: ${err.message}`);
    }
}

// Send Chat message (DeepSeek chat)
async function sendMessage() {
    const input = document.getElementById('chat-input');
    const text = input.value.stripOrEmpty();
    if (!text) return;

    // Clear input box
    input.value = '';
    input.style.height = '24px';

    const messagesContainer = document.getElementById('chat-messages');

    // Append User Message
    const userMsg = document.createElement('div');
    userMsg.className = 'message user';
    userMsg.innerHTML = `<div class="message-content">${escapeHtml(text)}</div>`;
    messagesContainer.appendChild(userMsg);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;

    // Create Assistant Message container
    const assistantMsg = document.createElement('div');
    assistantMsg.className = 'message assistant';
    messagesContainer.appendChild(assistantMsg);

    const thinkingMode = document.getElementById('toggle-thinking').checked;
    
    // Auth header fallback (checks saved generated key first, then admin key)
    const activeUserKey = localStorage.getItem('nexus_active_user_key') || localStorage.getItem('nexus_admin_key') || '';

    try {
        const headers = {
            'Content-Type': 'application/json'
        };
        if (activeUserKey) {
            headers['X-API-Key'] = activeUserKey;
        }

        // Send streaming request or normal request depending on backend
        const res = await fetch('/chat', {
            method: 'POST',
            headers,
            body: JSON.stringify({
                message: text,
                thinking: thinkingMode
            })
        });

        if (!res.ok) {
            if (res.status === 401) {
                assistantMsg.innerHTML = `<div class="message-content" style="color: var(--danger-color);">⚠️ Unauthorized. Please generate an API Key or enter an Admin Key in the API Keys tab.</div>`;
            } else if (res.status === 429) {
                assistantMsg.innerHTML = `<div class="message-content" style="color: var(--danger-color);">⚠️ Rate limit exceeded. Please wait a minute before sending another message.</div>`;
            } else {
                const err = await res.json().catch(() => ({}));
                assistantMsg.innerHTML = `<div class="message-content" style="color: var(--danger-color);">⚠️ Error (${res.status}): ${err.detail || res.statusText}</div>`;
            }
            return;
        }

        const data = await res.json();
        
        // Render thinking trace if present
        let htmlContent = '';
        if (thinkingMode && data.thinking) {
            htmlContent += `
                <div class="thinking-trace">
                    <div class="thinking-header" onclick="toggleThinkingBlock(this)">
                        <span>⚡ Thought process</span>
                        <span class="collapse-arrow">▼</span>
                    </div>
                    <div class="thinking-content">${escapeHtml(data.thinking)}</div>
                </div>
            `;
        }

        htmlContent += `<div class="message-content">${escapeHtml(data.response)}</div>`;
        assistantMsg.innerHTML = htmlContent;
        messagesContainer.scrollTop = messagesContainer.scrollHeight;

    } catch (err) {
        assistantMsg.innerHTML = `<div class="message-content" style="color: var(--danger-color);">⚠️ Network error: ${err.message}</div>`;
    }
}

// Collapsible thinking traces
function toggleThinkingBlock(headerEl) {
    const parent = headerEl.parentElement;
    parent.classList.toggle('collapsed');
    const arrow = headerEl.querySelector('.collapse-arrow');
    arrow.textContent = parent.classList.contains('collapsed') ? '▶' : '▼';
}

// Utilities
function escapeHtml(text) {
    if (!text) return '';
    return text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

function copyToClipboard(elementId) {
    const input = document.getElementById(elementId);
    input.select();
    input.setSelectionRange(0, 99999);
    navigator.clipboard.writeText(input.value);
    alert('Copied to clipboard!');
}
