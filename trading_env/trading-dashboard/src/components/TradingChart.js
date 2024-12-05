import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';
import { Paper, Typography } from '@mui/material';

const TradingChart = ({ data }) => {
    return (
        <Paper sx={{ padding: 2 }}>
            <Typography variant="h6">Portfolio Performance</Typography>
            <LineChart width={800} height={400} data={data}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="value" stroke="#8884d8" />
                <Line type="monotone" dataKey="prediction" stroke="#82ca9d" />
            </LineChart>
        </Paper>
    );
};

export default TradingChart; 