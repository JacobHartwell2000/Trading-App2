import axios from 'axios';

const API_BASE_URL = 'http://127.0.0.1:5000';

// Update axios instance config
const api = axios.create({
    baseURL: API_BASE_URL,
    timeout: 5000,
    headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    },
    withCredentials: false  // Important: set this to false
});

// Add request interceptor for debugging
api.interceptors.request.use(
    config => {
        console.log('Starting Request:', {
            url: config.url,
            method: config.method,
            headers: config.headers
        });
        return config;
    },
    error => {
        console.error('Request Error:', error);
        return Promise.reject(error);
    }
);

// Add response interceptor for error handling
api.interceptors.response.use(
    response => {
        console.log('Response:', response.data);
        return response;
    },
    error => {
        if (error.response) {
            console.error('API Error:', {
                status: error.response.status,
                data: error.response.data,
                headers: error.response.headers
            });
        } else if (error.request) {
            console.error('Network Error:', error.request);
        } else {
            console.error('Request Config Error:', error.message);
        }
        return Promise.reject(error);
    }
);

// Update health check endpoint to match backend response structure
export const checkHealth = async () => {
    return withRetry(async () => {
        try {
            const response = await api.get('/api/health');
            return response.data.status === 'success' && response.data.data.bot_initialized;
        } catch (error) {
            console.error('Health check failed:', error);
            return false;
        }
    }, 3, 2000);
};

// Add account status endpoint
export const fetchAccountStatus = async () => {
    try {
        console.log('Fetching account status from:', `${API_BASE_URL}/api/account-status`);
        const response = await api.get('/api/account-status');
        console.log('Raw Account Status Response:', response.data);
        
        if (response.data.status === 'success') {
            if (!response.data.data) {
                throw new Error('No data received from server');
            }
            return response.data.data;
        }
        throw new Error(response.data.message || 'Failed to fetch account status');
    } catch (error) {
        console.error('Failed to fetch account status:', error);
        console.error('Error details:', {
            message: error.message,
            response: error.response?.data,
            status: error.response?.status
        });
        return { error: error.message };
    }
};

// Trading status endpoint
export const fetchTradingStatus = async () => {
    try {
        const response = await api.get('/debug/bot-status');
        return response;
    } catch (error) {
        console.error('Error fetching trading status:', error);
        throw error;
    }
};

// Test connection endpoint
export const testConnection = async () => {
    try {
        const response = await api.get('/test');
        return response.data;
    } catch (error) {
        console.error('Test connection failed:', error);
        throw error;
    }
};

// Update activity logs endpoint to match error handling pattern
export const fetchActivityLogs = async () => {
    try {
        const response = await api.get('/api/activity-logs');
        if (response.data.status === 'success') {
            return response.data.data;
        }
        throw new Error(response.data.message || 'Failed to fetch activity logs');
    } catch (error) {
        console.error('Failed to fetch activity logs:', error);
        throw error;
    }
};

// Update positions endpoint to match error handling pattern
export const fetchPositions = async () => {
    try {
        const response = await api.get('/api/positions');
        if (response.data.status === 'success') {
            return response.data.data;
        }
        throw new Error(response.data.message || 'Failed to fetch positions');
    } catch (error) {
        console.error('Failed to fetch positions:', error);
        throw error;
    }
};

// Add getPrediction endpoint
export const getPrediction = async (symbol) => {
    try {
        const response = await api.get(`/prediction/${symbol}`);
        return response;
    } catch (error) {
        console.error('Failed to fetch prediction:', error);
        return { data: null }; // Return null to prevent UI breaks
    }
};

// Utility function to handle retries
const withRetry = async (fn, retries = 3, delay = 1000) => {
    for (let i = 0; i < retries; i++) {
        try {
            return await fn();
        } catch (error) {
            if (i === retries - 1) throw error;
            await new Promise(resolve => setTimeout(resolve, delay));
            console.log(`Retrying... Attempt ${i + 2}/${retries}`);
        }
    }
};

export default api; 