import React, { useEffect, useState } from 'react';
import { Paper, Grid, Typography, Box, CircularProgress } from '@mui/material';
import { fetchAccountStatus } from '../services/api';

const AccountStatus = () => {
    const [accountData, setAccountData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        const loadAccountData = async () => {
            try {
                setLoading(true);
                const data = await fetchAccountStatus();
                console.log('Received account data:', data);
                setAccountData(data);
                setError(null);
            } catch (err) {
                console.error('Error loading account data:', err);
                setError(err.message);
            } finally {
                setLoading(false);
            }
        };

        loadAccountData();
        // Refresh every 30 seconds
        const interval = setInterval(loadAccountData, 30000);
        return () => clearInterval(interval);
    }, []);

    if (loading) {
        return (
            <Paper sx={{ p: 3, borderRadius: 2 }}>
                <Typography variant="h6" sx={{ mb: 3, color: 'primary.main', fontWeight: 600 }}>
                    Account Status
                </Typography>
                <Box display="flex" justifyContent="center" alignItems="center">
                    <CircularProgress />
                </Box>
            </Paper>
        );
    }

    if (error || !accountData || (!accountData.equity && !accountData.data?.equity)) {
        // Check if we're receiving bot status instead of account data
        if (accountData?.bot_exists !== undefined) {
            return (
                <Paper sx={{ p: 3, borderRadius: 2 }}>
                    <Typography variant="h6" sx={{ mb: 3, color: 'warning.main', fontWeight: 600 }}>
                        Account Status Unavailable
                    </Typography>
                    <Typography color="warning.main">
                        Bot is running but account data is not yet available. Please check Alpaca API configuration.
                    </Typography>
                </Paper>
            );
        }

        return (
            <Paper sx={{ p: 3, borderRadius: 2 }}>
                <Typography variant="h6" sx={{ mb: 3, color: 'error.main', fontWeight: 600 }}>
                    Account Status Error
                </Typography>
                <Typography color="error">
                    {error || 'Unable to fetch account data. Please check API configuration.'}
                </Typography>
            </Paper>
        );
    }

    const formatCurrency = (value) => {
        if (value === undefined || value === null || isNaN(value)) return '$0.00';
        return new Intl.NumberFormat('en-US', { 
            style: 'currency', 
            currency: 'USD' 
        }).format(value);
    };

    return (
        <Paper sx={{ p: 3, borderRadius: 2 }}>
            <Typography variant="h6" sx={{ mb: 3, color: 'primary.main', fontWeight: 600 }}>
                Account Status
            </Typography>
            <Grid container spacing={3}>
                {[
                    { label: 'Equity', value: accountData.equity },
                    { label: 'Buying Power', value: accountData.buying_power },
                    { label: 'Cash', value: accountData.cash },
                    { label: 'Portfolio Value', value: accountData.portfolio_value }
                ].map((item) => (
                    <Grid item xs={12} sm={6} md={3} key={item.label}>
                        <Box sx={{ p: 2, bgcolor: 'rgba(33, 150, 243, 0.1)', borderRadius: 1 }}>
                            <Typography variant="subtitle2" color="text.secondary">
                                {item.label}
                            </Typography>
                            <Typography variant="h6" sx={{ mt: 1, color: 'primary.main' }}>
                                {formatCurrency(item.value)}
                            </Typography>
                        </Box>
                    </Grid>
                ))}
            </Grid>
        </Paper>
    );
};

export default AccountStatus; 