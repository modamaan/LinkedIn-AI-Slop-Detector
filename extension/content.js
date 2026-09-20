// LinkedIn AI Slop Detector Content Script

// IMPORTANT: We will replace this URL once the Modal training finishes and we deploy the API endpoint.
const LAYA_API_URL = "https://<YOUR_MODAL_USERNAME>--laya-slop-api-fastapi-app.modal.run/predict";

// Keep track of processed posts to avoid duplicate API calls
const processedPosts = new WeakSet();

// Main observer to watch for new posts as the user scrolls
const observer = new MutationObserver((mutations) => {
    for (const mutation of mutations) {
        if (mutation.addedNodes.length) {
            findAndProcessPosts();
        }
    }
});

function init() {
    console.log("🤖 Laya Slop Detector: Initializing...");
    findAndProcessPosts();
    observer.observe(document.body, { childList: true, subtree: true });
}

function findAndProcessPosts() {
    // LinkedIn is actively obfuscating their CSS classes (A/B testing new layouts).
    // To be 100% robust, we completely ignore classes and just look for the text itself!
    
    const textNodes = [];
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, null, false);
    
    let node;
    while(node = walker.nextNode()) {
        const text = node.textContent.trim();
        // Slop is wordy. We look for continuous text nodes > 100 chars.
        if (text.length > 100) {
            const parentName = node.parentElement.tagName.toLowerCase();
            if (parentName !== 'script' && parentName !== 'style') {
                textNodes.push(node);
            }
        }
    }
    
    if (textNodes.length === 0) return; // Wait for content to load

    textNodes.forEach((textNode) => {
        const container = textNode.parentElement;
        if (processedPosts.has(container)) return;
        
        // Find the "card" wrapper to attach the badge to.
        // We look for any data-urn (LinkedIn's tracking ID) or just go up 4 levels to wrap the whole post area.
        let postWrapper = container.closest('[data-urn], [data-id]');
        if (!postWrapper) {
            postWrapper = container.parentElement?.parentElement?.parentElement || container;
        }
        
        // If we already badged this post wrapper (because it had multiple paragraphs), skip it.
        if (processedPosts.has(postWrapper)) {
            processedPosts.add(container); // Mark this paragraph as seen too
            return;
        }
        
        const postText = postWrapper.innerText.trim();
        if (postText.length < 50) return;
        
        console.log(`🤖 Laya: Found post text! Length: ${postText.length}`);
        
        // Mark both the paragraph and the wrapper as processed
        processedPosts.add(container);
        processedPosts.add(postWrapper);
        
        // Inject loading badge directly into the text paragraph so it can't be clipped!
        const badge = createBadge();
        container.insertBefore(badge, container.firstChild);
        
        // Analyze the text
        analyzeText(postText, badge);
    });
}

function createBadge() {
    const badge = document.createElement('div');
    badge.className = 'slop-badge status-loading';
    badge.innerHTML = `
        <div class="slop-badge-dot"></div>
        <span>Analyzing...</span>
    `;
    return badge;
}

async function analyzeText(text, badgeElement) {
    try {
        // 1. Local Heuristic: Find all emojis in the text
        const emojiRegex = /\p{Emoji_Presentation}/gu;
        const emojis = text.match(emojiRegex);
        const emojiCount = emojis ? emojis.length : 0;
        
        let emojiProbability = 0.01;
        if (emojiCount >= 4) {
            emojiProbability = 0.99; // Definitely Slop
        } else if (emojiCount >= 2) {
            emojiProbability = 0.60; // Mixed
        }
        
        console.log(`🤖 Laya: Local Emoji Heuristic suggests ${emojiProbability * 100}% Slop`);

        // 2. Cloud ML Model: Send request to our deployed Laya model
        const response = await fetch(LAYA_API_URL, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ text: text })
        });
        
        if (!response.ok) {
            throw new Error(`API Error: ${response.status}`);
        }
        
        const data = await response.json();
        const mlProbability = data.slop_probability; 
        console.log(`🤖 Laya: Modal ML Model suggests ${(mlProbability * 100).toFixed(1)}% Slop`);
        
        // 3. Ensemble: Combine them! 
        // We take the max. If either the ML model spots slop semantics OR it's flooded with emojis, it's slop!
        const finalProbability = Math.max(mlProbability, emojiProbability);
        
        updateBadgeState(badgeElement, finalProbability);
        
    } catch (error) {
        console.error("Laya API Error:", error);
        badgeElement.className = 'slop-badge status-loading';
        badgeElement.innerHTML = `<div class="slop-badge-dot"></div><span>API Error</span>`;
        badgeElement.setAttribute('data-tooltip', 'Make sure the Modal API is running');
    }
}

function updateBadgeState(badge, probability) {
    // 1. Remove loading state
    badge.className = 'slop-badge';
    
    // 2. Determine state based on probability
    let label = "";
    let stateClass = "";
    let tooltip = `AI Slop Probability: ${(probability * 100).toFixed(1)}%`;
    
    if (probability < 0.3) {
        label = "Human";
        stateClass = "status-human";
    } else if (probability < 0.7) {
        label = "Mixed";
        stateClass = "status-mixed";
        tooltip += " - Contains typical LinkedIn templates";
    } else {
        label = "AI Slop";
        stateClass = "status-slop";
        tooltip += " - Highly likely to be AI generated";
    }
    
    // 3. Update DOM
    badge.classList.add(stateClass);
    badge.innerHTML = `
        <div class="slop-badge-dot"></div>
        <span>${label}</span>
        ${probability >= 0.7 ? `<span style="opacity: 0.6; font-size: 9px; margin-left: 2px;">${(probability * 100).toFixed(0)}%</span>` : ''}
    `;
    badge.setAttribute('data-tooltip', tooltip);
    
    // Optional: Auto-collapse severe slop
    // if (probability > 0.9) {
    //    const parent = badge.closest('.feed-shared-update-v2');
    //    if (parent) parent.style.opacity = '0.3'; 
    // }
}

// Start observing once the page loads
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
