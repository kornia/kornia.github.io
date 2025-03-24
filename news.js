// Global variables
let currentPage = 1;
let itemsPerPage = 10;
let activeFilters = new Set(['all']);

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
        <article class="news-item" data-tags="${item.tags.join(' ')}">
            <div class="news-item-header">
                <div class="news-date">${item.date}</div>
                <div class="news-tags">
                    ${item.tags.map(tag => `<span class="tag">#${tag}</span>`).join(' ')}
                </div>
            </div>
            <h2>${item.title}</h2>
            <div class="news-item-content">
                <p>${item.content}</p>
                <a href="${item.link}" class="button secondary">Read More</a>
            </div>
        </article>
    `;
}

// Filter news items based on active filters
function filterNews() {
    return newsData.items.filter(item => {
        if (activeFilters.has('all')) return true;
        return item.tags.some(tag => activeFilters.has(tag));
    });
}

// Update visible items based on current page and filters
function updateVisibleItems() {
    const filteredItems = filterNews();
    const startIndex = (currentPage - 1) * itemsPerPage;
    const endIndex = startIndex + itemsPerPage;
    const itemsToShow = filteredItems.slice(startIndex, endIndex);
    
    const newsList = document.getElementById('newsList');
    newsList.innerHTML = itemsToShow.map(item => createNewsItemHTML(item)).join('');
    
    updatePagination(filteredItems.length);
}

// Update pagination
function updatePagination(totalItems) {
    const totalPages = Math.ceil(totalItems / itemsPerPage);
    const pagination = document.querySelector('.pagination');
    
    let paginationHTML = `
        <button class="pagination-btn" id="prevPage" ${currentPage === 1 ? 'disabled' : ''}>Previous</button>
    `;
    
    // Add page numbers
    for (let i = 1; i <= totalPages; i++) {
        if (
            i === 1 || // First page
            i === totalPages || // Last page
            (i >= currentPage - 1 && i <= currentPage + 1) // Pages around current
        ) {
            paginationHTML += `
                <button class="pagination-number ${i === currentPage ? 'active' : ''}" data-page="${i}">${i}</button>
            `;
        } else if (
            i === currentPage - 2 || // Before ellipsis
            i === currentPage + 2 // After ellipsis
        ) {
            paginationHTML += '<span class="pagination-ellipsis">...</span>';
        }
    }
    
    paginationHTML += `
        <button class="pagination-btn" id="nextPage" ${currentPage === totalPages ? 'disabled' : ''}>Next</button>
    `;
    
    pagination.innerHTML = paginationHTML;
}

// Go to specific page
function goToPage(page) {
    currentPage = page;
    updateVisibleItems();
}

// Event Listeners
document.addEventListener('DOMContentLoaded', function() {
    // Initialize hashtag filters
    createHashtagFilters();
    
    const hashtagFilters = document.querySelectorAll('.hashtag-filter');
    const itemsPerPageSelect = document.getElementById('itemsPerPage');
    
    // Event Listeners
    hashtagFilters.forEach(filter => {
        filter.onclick = function() {
            const tag = this.dataset.tag;
            
            if (tag === 'all') {
                // If clicking 'all', clear other filters
                hashtagFilters.forEach(f => f.classList.remove('active'));
                this.classList.add('active');
                activeFilters.clear();
                activeFilters.add('all');
            } else {
                // If clicking a specific tag
                const allFilter = document.querySelector('[data-tag="all"]');
                allFilter.classList.remove('active');
                activeFilters.delete('all'); // Remove 'all' from active filters
                
                this.classList.toggle('active');
                
                if (this.classList.contains('active')) {
                    activeFilters.add(tag);
                } else {
                    activeFilters.delete(tag);
                }
                
                // If no filters are active, default to 'all'
                if (activeFilters.size === 0) {
                    allFilter.classList.add('active');
                    activeFilters.add('all');
                }
            }
            
            currentPage = 1;
            updateVisibleItems();
        };
    });

    itemsPerPageSelect.onchange = function() {
        itemsPerPage = parseInt(this.value);
        currentPage = 1;
        updateVisibleItems();
    };
    
    // Pagination click handlers
    document.addEventListener('click', (e) => {
        if (e.target.id === 'prevPage' && currentPage > 1) {
            goToPage(currentPage - 1);
        } else if (e.target.id === 'nextPage') {
            const totalPages = Math.ceil(filterNews().length / itemsPerPage);
            if (currentPage < totalPages) {
                goToPage(currentPage + 1);
            }
        } else if (e.target.classList.contains('pagination-number')) {
            goToPage(parseInt(e.target.dataset.page));
        }
    });

    // Initialize
    updateVisibleItems();
}); 