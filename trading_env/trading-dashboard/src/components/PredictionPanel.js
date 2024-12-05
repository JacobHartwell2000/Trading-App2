import React, { useState } from 'react';
import { Paper, Typography, TextField, Button, Box } from '@mui/material';
import { getPrediction } from '../services/api';

const PredictionPanel = () => {
    const [symbol, setSymbol] = useState('');
    const [prediction, setPrediction] = useState(null);

    const handlePredict = async () => {
        try {
            const result = await getPrediction(symbol);
            setPrediction(result.data);
        } catch (error) {
            console.error('Error getting prediction:', error);
        }
    };

    return (
        <Paper sx={{ p: 3, borderRadius: 2, height: '100%' }}>
            <Typography variant="h6" sx={{ mb: 3, color: 'primary.main', fontWeight: 600 }}>
                Market Prediction
            </Typography>
            <Box sx={{ mt: 2 }}>
                <TextField
                    label="Symbol"
                    value={symbol}
                    onChange={(e) => setSymbol(e.target.value)}
                    fullWidth
                    variant="outlined"
                    sx={{
                        '& .MuiOutlinedInput-root': {
                            '&:hover fieldset': {
                                borderColor: 'primary.main',
                            },
                        },
                    }}
                />
                <Button
                    variant="contained"
                    onClick={handlePredict}
                    fullWidth
                    sx={{ 
                        mt: 2,
                        height: 48,
                        textTransform: 'none',
                        fontWeight: 600
                    }}
                >
                    Get Prediction
                </Button>
            </Box>
            {prediction && (
                <Box sx={{ 
                    mt: 3,
                    p: 2,
                    bgcolor: 'rgba(33, 150, 243, 0.1)',
                    borderRadius: 1
                }}>
                    <Typography variant="h6" color="primary">
                        {prediction.prediction === 1 ? 'Buy' : 'Sell'}
                    </Typography>
                    <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                        Confidence: {(prediction.confidence * 100).toFixed(2)}%
                    </Typography>
                </Box>
            )}
        </Paper>
    );
};

export default PredictionPanel; 