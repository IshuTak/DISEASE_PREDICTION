// Main JavaScript file
$(document).ready(function() {
    // Initialize Select2
    initializeSelect2();
    
    // Initialize theme
    initializeTheme();
    
    // Setup event listeners
    setupEventListeners();
    
    // Initialize mobile menu
    initializeMobileMenu();
});


function initializeSelect2() {
    $('#symptoms').select2({
        placeholder: 'Select symptoms',
        multiple: true,
        width: '100%',
        templateResult: formatSymptom,
        templateSelection: formatSymptomSelection
    });
}


function formatSymptom(symptom) {
    if (!symptom.id) return symptom.text;
    return $(`<span><i class="fas fa-check-circle"></i> ${symptom.text}</span>`);
}

function formatSymptomSelection(symptom) {
    return symptom.text;
}

// Theme initialization and handling
function initializeTheme() {
    const savedTheme = localStorage.getItem('theme') || 'light';
    document.documentElement.setAttribute('data-theme', savedTheme);
    
    $('#theme-toggle').on('click', function() {
        const currentTheme = document.documentElement.getAttribute('data-theme');
        const newTheme = currentTheme === 'light' ? 'dark' : 'light';
        
        document.documentElement.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);
        
        // Update icon
        $(this).find('i').toggleClass('fa-moon fa-sun');
    });
}

// Setup all event listeners
function setupEventListeners() {
    // Form submission
    $('#symptomForm').on('submit', handleFormSubmission);
    
    // Symptom search
    $('#symptom-search').on('input', handleSymptomSearch);
    
    // Category buttons
    $('.category-btn').on('click', handleCategoryFilter);
    
    // Scroll events
    $(window).on('scroll', handleScroll);
}

// Handle form submission
async function handleFormSubmission(e) {
    e.preventDefault();
    
    const symptoms = $('#symptoms').val();
    
    if (!symptoms.length) {
        showAlert('Please select at least one symptom', 'warning');
        return;
    }
    
    try {
        showLoading();
        
        const response = await $.ajax({
            url: '/predict',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ symptoms: symptoms })
        });
        
        hideLoading();
        
        if (response.success) {
            displayPredictions(response.predictions, symptoms);
            scrollToResults();
        } else {
            showAlert(response.error || 'Prediction failed', 'error');
        }
        
    } catch (error) {
        hideLoading();
        showAlert('Error making prediction', 'error');
        console.error('Prediction error:', error);
    }
}

// Display predictions
function displayPredictions(predictions, selectedSymptoms) {
    const resultsDiv = $('#results');
    resultsDiv.empty();
    
    // Add selected symptoms section
    resultsDiv.append(`
        <div class="selected-symptoms">
            <h3>Selected Symptoms:</h3>
            <p>${selectedSymptoms.map(s => s.replace(/_/g, ' ')).join(', ')}</p>
        </div>
    `);
    
    // Add predictions
    predictions.forEach((pred, index) => {
        const probability = (pred.probability * 100).toFixed(2);
        const confidenceClass = getConfidenceClass(pred.confidence_level);
        
        const html = `
            <div class="prediction ${confidenceClass}">
                <h3>${index + 1}. ${pred.disease}</h3>
                <div class="probability-section">
                    <div class="probability-bar-container">
                        <div class="probability-bar" style="width: ${probability}%"></div>
                    </div>
                    <p class="probability-text">Probability: ${probability}%</p>
                    <p class="confidence">Confidence Level: ${pred.confidence_level}</p>
                </div>

                <div class="info-section">
                    <div class="info-box">
                        <h4>Common Symptoms</h4>
                        <p>${pred.symptoms.map(s => s.replace(/_/g, ' ')).join(', ')}</p>
                    </div>

                    <div class="info-box">
                        <h4>Description</h4>
                        <p>${pred.description}</p>
                    </div>

                    <div class="info-box">
                        <h4>Precautions</h4>
                        <ul>
                            ${Array.isArray(pred.precautions) ? 
                                pred.precautions.map(p => `<li>${p}</li>`).join('') : 
                                '<li>No precautions available</li>'}
                        </ul>
                    </div>
                </div>
            </div>
        `;
        
        resultsDiv.append(html);
    });
    
    resultsDiv.removeClass('hidden');
    
    // Animate probability bars
    setTimeout(() => {
        $('.probability-bar').each(function() {
            $(this).css('width', $(this).parent().width() * parseFloat($(this).css('width')) / 100);
        });
    }, 100);
}

// Handle symptom search
function handleSymptomSearch() {
    const searchTerm = $(this).val().toLowerCase();
    
    $('#symptoms option').each(function() {
        const symptom = $(this).text().toLowerCase();
        $(this).toggle(symptom.includes(searchTerm));
    });
}

// Handle category filter
function handleCategoryFilter() {
    $('.category-btn').removeClass('active');
    $(this).addClass('active');
    
    const category = $(this).data('category');
    filterSymptomsByCategory(category);
}

// Filter symptoms by category
function filterSymptomsByCategory(category) {
    if (category === 'all') {
        $('#symptoms option').show();
        return;
    }
    
    $('#symptoms option').each(function() {
        const symptom = $(this).text().toLowerCase();
        const shouldShow = SYMPTOM_CATEGORIES[category].some(s => 
            symptom.includes(s.toLowerCase())
        );
        $(this).toggle(shouldShow);
    });
}

// Utility functions
function showLoading() {
    $('.loading-overlay').removeClass('hidden');
}

function hideLoading() {
    $('.loading-overlay').addClass('hidden');
}

function showAlert(message, type) {
    const alertDiv = $(`
        <div class="alert alert-${type}">
            ${message}
            <button type="button" class="close-alert">&times;</button>
        </div>
    `);
    
    $('.container').prepend(alertDiv);
    
    setTimeout(() => {
        alertDiv.fadeOut(() => alertDiv.remove());
    }, 5000);
}

function getConfidenceClass(confidence) {
    switch(confidence.toLowerCase()) {
        case 'high':
            return 'high-confidence';
        case 'medium':
            return 'medium-confidence';
        default:
            return 'low-confidence';
    }
}

function scrollToResults() {
    $('html, body').animate({
        scrollTop: $('#results').offset().top - 100
    }, 1000);
}

function scrollToPredictor() {
    $('html, body').animate({
        scrollTop: $('#predictor').offset().top - 100
    }, 1000);
}

// Handle scroll events
function handleScroll() {
    const scrollTop = $(window).scrollTop();
    
    // Add shadow to navbar on scroll
    if (scrollTop > 50) {
        $('.navbar').addClass('scrolled');
    } else {
        $('.navbar').removeClass('scrolled');
    }
    
    // Animate elements on scroll
    $('.animate-on-scroll').each(function() {
        const elementTop = $(this).offset().top;
        const elementVisible = 150;
        
        if (scrollTop + window.innerHeight > elementTop + elementVisible) {
            $(this).addClass('animated');
        }
    });
}

// Mobile menu handling
function initializeMobileMenu() {
    $('.mobile-menu-btn').on('click', function() {
        $('.mobile-menu').addClass('active');
    });
    
    $('.close-menu').on('click', function() {
        $('.mobile-menu').removeClass('active');
    });
    
    // Close menu on link click
    $('.mobile-menu a').on('click', function() {
        $('.mobile-menu').removeClass('active');
    });
}

// Symptom categories for filtering
const SYMPTOM_CATEGORIES = {
    common: ['fever', 'headache', 'fatigue', 'pain'],
    pain: ['pain', 'ache', 'sore', 'hurt'],
    respiratory: ['cough', 'breath', 'chest', 'throat'],
    digestive: ['stomach', 'nausea', 'vomit', 'diarrhea']
};

// Add smooth scrolling for all anchor links
$(document).on('click', 'a[href^="#"]', function(e) {
    e.preventDefault();
    
    const target = $(this.hash);
    if (target.length) {
        $('html, body').animate({
            scrollTop: target.offset().top - 80
        }, 1000);
    }
});