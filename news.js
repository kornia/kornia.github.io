// Global variables
let currentPage = 1;
let itemsPerPage = 4;
let activeFilters = new Set(['all']);
let loadedCount = 0;
let isLoading = false;

// Create hashtag filter buttons
function createHashtagFilters() {
    const hashtagFilters = document.getElementById('hashtagFilters');
    hashtagFilters.innerHTML = newsData.tags.map(tag => `
        <button class="hashtag-filter ${tag === 'all' ? 'active' : ''}" data-tag="${tag}">
            ${tag === 'all' ? 'All' : `#${tag}`}
        </button>
    `).join('');
}

// Create news item HTML
function createNewsItemHTML(item) {
    return `
        <div class="col-12 col-md-6 mb-4" style="min-width: 350px; max-width: 400px; margin-right: 1.5rem;">
            <article class="news-item" data-tags="${item.tags.join(' ')}" style="height: 400px; display: flex; flex-direction: column;">
                <div class="news-item-header">
                    <div class="news-date">${item.date}</div>
                    <div class="news-tags">
                        ${item.tags.map(tag => `<span class="tag">#${tag}</span>`).join(' ')}
                    </div>
                </div>
                <h2 style="font-size: 1.25rem; margin-bottom: 1rem;">${item.title}</h2>
                <div class="news-item-content" style="flex: 1; overflow: hidden;">
                    <div class="news-text-collapse" style="max-height: 120px; overflow: hidden;">
                        <p style="margin-bottom: 0;">${item.content}</p>
                    </div>
                    <button class="btn btn-link btn-sm p-0 mt-2 text-primary" onclick="showModal(this)" data-item-index="${newsData.items.indexOf(item)}" style="display: inline-block;">Expand</button>
                </div>
                <div class="mt-auto pt-3">
                    ${item.link ? `<a href="${item.link}" target="_blank">Read More</a>` : ''}
                </div>
            </article>
        </div>
    `;
}

// Filter news items based on active filters
function filterNews() {
    return newsData.items.filter(item => {
        if (activeFilters.has('all')) return true;
        return item.tags.some(tag => activeFilters.has(tag));
    });
}

// Update visible items based on filters
function updateVisibleItems() {
    const filteredItems = filterNews();
    const newsList = document.getElementById('newsList');
    newsList.innerHTML = filteredItems.map(item => createNewsItemHTML(item)).join('');
    
    // Check for text overflow and add expand functionality
    setTimeout(() => {
        const newsItems = document.querySelectorAll('.news-item');
        newsItems.forEach((item, index) => {
            const textContainer = item.querySelector('.news-text-collapse');
            const expandBtn = item.querySelector('.btn-link');
            
            // Set a fixed height for the text container
            textContainer.style.maxHeight = '120px';
            textContainer.style.overflow = 'hidden';
            
            // Check if content is actually overflowing
            const isOverflowing = textContainer.scrollHeight > textContainer.clientHeight;
            
            if (isOverflowing) {
                expandBtn.style.display = 'inline-block';
            } else {
                expandBtn.style.display = 'none';
            }
            
            // For debugging - show expand button for all items with content
            if (textContainer.textContent.trim().length > 100) {
                expandBtn.style.display = 'inline-block';
            }
        });
    }, 200);
}

// Show modal with full news content
function showModal(button) {
    const itemIndex = parseInt(button.getAttribute('data-item-index'));
    const item = newsData.items[itemIndex];
    
    // Populate modal content
    document.getElementById('newsModalLabel').textContent = item.title;
    document.querySelector('.news-modal-date').textContent = item.date;
    document.querySelector('.news-modal-tags').innerHTML = item.tags.map(tag => `<span class="tag">#${tag}</span>`).join(' ');
    document.querySelector('.news-modal-content').innerHTML = item.content;
    
    // Handle Read More button
    const readMoreContainer = document.getElementById('newsModalReadMore');
    if (item.link) {
        readMoreContainer.innerHTML = `<a href="${item.link}" class="btn btn-primary" target="_blank">Read More</a>`;
    } else {
        readMoreContainer.innerHTML = '';
    }
    
    // Show modal
    const modal = new bootstrap.Modal(document.getElementById('newsModal'));
    modal.show();
}

// Event Listeners

document.addEventListener('DOMContentLoaded', function() {
    // Initialize hashtag filters
    createHashtagFilters();
    const hashtagFilters = document.querySelectorAll('.hashtag-filter');
    const itemsPerPageSelect = document.getElementById('itemsPerPage');

    // Left/Right slider controls
    document.getElementById('newsLeft').onclick = function() {
        document.getElementById('newsList').scrollBy({ left: -400, behavior: 'smooth' });
    };
    
    document.getElementById('newsRight').onclick = function() {
        document.getElementById('newsList').scrollBy({ left: 400, behavior: 'smooth' });
    };

    hashtagFilters.forEach(filter => {
        filter.onclick = function() {
            const tag = this.dataset.tag;
            if (tag === 'all') {
                hashtagFilters.forEach(f => f.classList.remove('active'));
                this.classList.add('active');
                activeFilters.clear();
                activeFilters.add('all');
            } else {
                const allFilter = document.querySelector('[data-tag="all"]');
                allFilter.classList.remove('active');
                activeFilters.delete('all');
                this.classList.toggle('active');
                if (this.classList.contains('active')) {
                    activeFilters.add(tag);
                } else {
                    activeFilters.delete(tag);
                }
                if (activeFilters.size === 0) {
                    allFilter.classList.add('active');
                    activeFilters.add('all');
                }
            }
            updateVisibleItems();
        };
    });

    itemsPerPageSelect.onchange = function() {
        itemsPerPage = parseInt(this.value);
        updateVisibleItems();
    };

    // Initialize
    updateVisibleItems();
}); 