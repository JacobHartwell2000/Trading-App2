import React, { useState, useEffect } from 'react';
import { Grid, Paper, Typography, CircularProgress, Box } from '@mui/material';
import AccountStatus from './AccountStatus';
import PositionsTable from './PositionsTable';
import TradingChart from './TradingChart';
import PredictionPanel from './PredictionPanel';
import AIChat from './AIChat';
import ActivityLog from './ActivityLog';
import { fetchTradingStatus, fetchPositions, checkHealth } from '../services/api';

const Dashboard = () => {
    const [accountData, setAccountData] = useState(null);
    const [positions, setPositions] = useState([]);
    const [error, setError] = useState(null);
    const [loading, setLoading] = useState(true);
    const [activities, setActivities] = useState([]);
    const [connectionStatus, setConnectionStatus] = useState('connecting');

    useEffect(() => {
        let mounted = true;
        let retryTimeout;
        let retryCount = 0;
        const maxRetries = 5;
        
        const checkConnection = async () => {
            try {
                const isHealthy = await checkHealth();
                if (mounted) {
                    if (isHealthy) {
                        setConnectionStatus('connected');
                        retryCount = 0;
                    } else {
                        handleConnectionFailure();
                    }
                }
            } catch (error) {
                if (mounted) {
                    handleConnectionFailure();
                }
            }
        };

        const handleConnectionFailure = () => {
            setConnectionStatus('disconnected');
            retryCount++;
            if (retryCount < maxRetries) {
                retryTimeout = setTimeout(checkConnection, 5000);
            }
        };

        // Initial check
        checkConnection();

        // Regular health checks
        const healthInterval = setInterval(checkConnection, 30000);

        return () => {
            mounted = false;
            clearTimeout(retryTimeout);
            clearInterval(healthInterval);
        };
    }, []);

    useEffect(() => {
        let mounted = true;
        
        const fetchData = async () => {
            try {
                setLoading(true);
                const [statusData, positionsData] = await Promise.all([
                    fetchTradingStatus(),
                    fetchPositions()
                ]);
                
                if (mounted) {
                    setAccountData(statusData);
                    setPositions(positionsData);
                    setError(null);
                }
            } catch (err) {
                if (mounted) {
                    setError(err.message);
                }
            } finally {
                if (mounted) {
                    setLoading(false);
                }
            }
        };

        fetchData();

        // Set up periodic refresh
        const refreshInterval = setInterval(fetchData, 60000); // Refresh every minute

        return () => {
            mounted = false;
            clearInterval(refreshInterval);
        };
    }, []);

    // Show connection status
    if (connectionStatus === 'disconnected') {
        return (
            <Box display="flex" justifyContent="center" alignItems="center" minHeight="200px">
                <Typography color="error">
                    Connection lost. Attempting to reconnect...
                </Typography>
            </Box>
        );
    }

    // Show loading spinner only for initial load
    if (loading && !accountData) {
        return (
            <Box display="flex" justifyContent="center" alignItems="center" minHeight="200px">
                <CircularProgress />
            </Box>
        );
    }

    if (error) {
        return <div>Error: {error}</div>;
    }

    return (
        <Grid container spacing={3}>
            <Grid item xs={12}>
                <AccountStatus accountData={accountData} />
            </Grid>
            <Grid item xs={12} md={8}>
                <TradingChart />
            </Grid>
            <Grid item xs={12} md={4}>
                <ActivityLog activities={activities} />
            </Grid>
            <Grid item xs={12}>
                <PositionsTable positions={positions} />
            </Grid>
            <Grid item xs={12} md={6}>
                <AIChat />
            </Grid>
            <Grid item xs={12} md={6}>
                <PredictionPanel />
            </Grid>
            <Grid item xs={12}>
                <Paper sx={{ p: 2, display: 'flex', flexDirection: 'column' }}>
                    <Typography variant="h6" gutterBottom component="div">
                        Market Sentiment Analysis
                    </Typography>
                    <Grid container spacing={2}>
                        <Grid item xs={12} md={4}>
                            <Paper sx={{ p: 2, bgcolor: 'background.default' }}>
                                <Typography variant="subtitle1">News Sentiment</Typography>
                                <Typography variant="h4" color={accountData?.data?.newsSentiment > 0 ? 'success.main' : 'error.main'}>
                                    {accountData?.data?.newsSentiment > 0 ? 'Bullish' : 'Bearish'}
                                </Typography>
                                <Typography variant="body2" color="text.secondary">
                                    Score: {accountData?.data?.newsSentiment?.toFixed(2) || 'N/A'}
                                </Typography>
                            </Paper>
                        </Grid>
                        <Grid item xs={12} md={4}>
                            <Paper sx={{ p: 2, bgcolor: 'background.default' }}>
                                <Typography variant="subtitle1">Social Media Sentiment</Typography>
                                <Typography variant="h4" color={accountData?.data?.socialSentiment > 0 ? 'success.main' : 'error.main'}>
                                    {accountData?.data?.socialSentiment > 0 ? 'Bullish' : 'Bearish'}
                                </Typography>
                                <Typography variant="body2" color="text.secondary">
                                    Score: {accountData?.data?.socialSentiment?.toFixed(2) || 'N/A'}
                                </Typography>
                            </Paper>
                        </Grid>
                        <Grid item xs={12} md={4}>
                            <Paper sx={{ p: 2, bgcolor: 'background.default' }}>
                                <Typography variant="subtitle1">Overall Market Sentiment</Typography>
                                <Typography variant="h4" color={accountData?.data?.overallSentiment > 0 ? 'success.main' : 'error.main'}>
                                    {accountData?.data?.overallSentiment > 0 ? 'Bullish' : 'Bearish'}
                                </Typography>
                                <Typography variant="body2" color="text.secondary">
                                    Score: {accountData?.data?.overallSentiment?.toFixed(2) || 'N/A'}
                                </Typography>
                            </Paper>
                        </Grid>
                    </Grid>
                </Paper>
            </Grid>
            <Grid item xs={12}>
                <Paper sx={{ p: 2, display: 'flex', flexDirection: 'column' }}>
                    <Typography variant="h6" gutterBottom component="div">
                        Market Regime Analysis
                    </Typography>
                    <Grid container spacing={2}>
                        <Grid item xs={12} md={4}>
                            <Paper sx={{ p: 2, bgcolor: 'background.default' }}>
                                <Typography variant="subtitle1">Current Regime</Typography>
                                <Typography variant="h4" color="primary">
                                    {accountData?.data?.marketRegime?.current || 'Unknown'}
                                </Typography>
                                <Typography variant="body2" color="text.secondary">
                                    Hurst Exponent: {accountData?.data?.marketRegime?.hurstExponent?.toFixed(3) || 'N/A'}
                                </Typography>
                            </Paper>
                        </Grid>
                        <Grid item xs={12} md={4}>
                            <Paper sx={{ p: 2, bgcolor: 'background.default' }}>
                                <Typography variant="subtitle1">Volatility Profile</Typography>
                                <Typography variant="h4" color={
                                    accountData?.data?.marketRegime?.volatilityRegime === 'low' ? 'success.main' :
                                    accountData?.data?.marketRegime?.volatilityRegime === 'high' ? 'error.main' :
                                    'warning.main'
                                }>
                                    {(accountData?.data?.marketRegime?.volatilityRegime || 'Unknown').toUpperCase()}
                                </Typography>
                                <Typography variant="body2" color="text.secondary">
                                    Volatility: {accountData?.data?.marketRegime?.volatility?.toFixed(3) || 'N/A'}
                                </Typography>
                            </Paper>
                        </Grid>
                        <Grid item xs={12} md={4}>
                            <Paper sx={{ p: 2, bgcolor: 'background.default' }}>
                                <Typography variant="subtitle1">Trading Parameters</Typography>
                                <Typography variant="body2" color="text.secondary">
                                    Window Size: {accountData?.data?.marketRegime?.parameters?.window_size || 'N/A'}
                                </Typography>
                                <Typography variant="body2" color="text.secondary">
                                    Volatility Threshold: {accountData?.data?.marketRegime?.parameters?.volatility_threshold?.toFixed(3) || 'N/A'}
                                </Typography>
                            </Paper>
                        </Grid>
                    </Grid>
                </Paper>
            </Grid>
        </Grid>
    );
};

export default Dashboard; 