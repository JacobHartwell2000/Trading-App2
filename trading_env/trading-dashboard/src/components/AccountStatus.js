import React from 'react';
import { Paper, Grid, Typography, Box, CircularProgress } from '@mui/material';

const AccountStatus = ({ accountData }) => {
    console.log('Raw AccountData:', accountData);

    if (!accountData) {
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

    if (accountData.error || !accountData.equity) {
        return (
            <Paper sx={{ p: 3, borderRadius: 2 }}>
                <Typography variant="h6" sx={{ mb: 3, color: 'error.main', fontWeight: 600 }}>
                    Account Status Error
                </Typography>   
                 
                 
                 
                 
                <Typography color="error">
                    {accountData.error || 'Unable to fetch account data. Please check API credentials.'}
                </Typography>
            </Paper>
        );
    }

    const { equity, buying_power, cash, portfolio_value } = accountData;

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
                    { label: 'Equity', value: equity },
                    { label: 'Buying Power', value: buying_power },
                    { label: 'Cash', value: cash },
                    { label: 'Portfolio Value', value: portfolio_value }
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