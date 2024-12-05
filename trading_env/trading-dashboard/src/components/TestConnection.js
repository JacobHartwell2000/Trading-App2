import React, { useEffect, useState } from 'react';
import { testConnection } from '../services/api';
import { Paper, Typography } from '@mui/material';

const TestConnection = () => {
    const [status, setStatus] = useState('Testing...');

    useEffect(() => {
        const test = async () => {
            try {
                const result = await testConnection();
                setStatus(`Connected! ${result.message}`);
            } catch (error) {
                setStatus(`Error: ${error.message}`);
            }
        };
        test();
    }, []);

    return (
        <Paper sx={{ p: 2, mb: 2 }}>
            <Typography>API Status: {status}</Typography>
        </Paper>
    );
};

export default TestConnection; 