import React, { useState, useEffect, useRef } from 'react';
import { Paper, Typography, List, ListItem, ListItemText, CircularProgress, Box } from '@mui/material';
import { fetchActivityLogs } from '../services/api';

const ActivityLog = () => {
    const [activities, setActivities] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const listRef = useRef(null);

    const scrollToBottom = () => {
        if (listRef.current) {
            listRef.current.scrollTop = listRef.current.scrollHeight;
        }
    };

    const fetchLogs = async () => {
        try {
            setLoading(true);
            const logs = await fetchActivityLogs();
            if (logs) {
                const formattedActivities = logs.map(activity => ({
                    ...activity,
                    timestamp: new Date(activity.timestamp).toLocaleString()
                }));
                setActivities(formattedActivities);
                setTimeout(scrollToBottom, 100);
            }
            setError(null);
        } catch (err) {
            console.error('Error fetching activity logs:', err);
            setError('Failed to load activity logs');
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchLogs();
        
        // Refresh logs every 10 seconds
        const interval = setInterval(fetchLogs, 10000);
        
        return () => clearInterval(interval);
    }, []);

    useEffect(() => {
        scrollToBottom();
    }, [activities]);

    if (loading && activities.length === 0) {
        return (
            <Paper sx={{ p: 2, display: 'flex', justifyContent: 'center' }}>
                <CircularProgress />
            </Paper>
        );
    }

    if (error) {
        return (
            <Paper sx={{ p: 2 }}>
                <Typography color="error">{error}</Typography>
            </Paper>
        );
    }

    return (
        <Paper 
            ref={listRef}
            sx={{ 
                p: 2, 
                maxHeight: 450, 
                minHeight: 400, 
                overflow: 'auto',
                display: 'flex',
                flexDirection: 'column'
            }}
        >
            <Typography variant="h6" gutterBottom>
                Activity Log
            </Typography>
            {activities.length === 0 ? (
                <Typography color="textSecondary">
                    No activities logged yet
                </Typography>
            ) : (
                <List dense sx={{ flex: 1 }}>
                    {activities.map((activity, index) => (
                        <ListItem 
                            key={index}
                            sx={{
                                borderLeft: '3px solid',
                                borderLeftColor: 
                                    activity.type === 'error' ? 'error.main' :
                                    activity.type === 'trade' ? 'success.main' :
                                    activity.type === 'analysis' ? 'info.main' :
                                    'grey.500',
                                mb: 1,
                                backgroundColor: 'background.paper',
                                '&:hover': {
                                    backgroundColor: 'action.hover',
                                }
                            }}
                        >
                            <ListItemText
                                primary={
                                    <Typography 
                                        component="pre"
                                        sx={{ 
                                            fontFamily: 'monospace',
                                            fontSize: '0.9rem',
                                            whiteSpace: 'pre-wrap',
                                            wordWrap: 'break-word'
                                        }}
                                    >
                                        {activity.message}
                                    </Typography>
                                }
                                secondary={
                                    <Typography variant="caption" color="textSecondary">
                                        {activity.timestamp}
                                    </Typography>
                                }
                            />
                        </ListItem>
                    ))}
                </List>
            )}
        </Paper>
    );
};

export default ActivityLog; 