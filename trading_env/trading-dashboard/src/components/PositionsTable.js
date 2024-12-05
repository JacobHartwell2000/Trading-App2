import React from 'react';
import {
    Paper,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    Typography
} from '@mui/material';

const PositionsTable = ({ positions }) => {
    return (
        <Paper sx={{ p: 3, borderRadius: 2 }}>
            <Typography variant="h6" sx={{ mb: 3, color: 'primary.main', fontWeight: 600 }}>
                Current Positions
            </Typography>
            <TableContainer>
                <Table>
                    <TableHead>
                        <TableRow>
                            <TableCell sx={{ fontWeight: 600, color: 'primary.main' }}>Symbol</TableCell>
                            <TableCell align="right" sx={{ fontWeight: 600, color: 'primary.main' }}>Quantity</TableCell>
                            <TableCell align="right" sx={{ fontWeight: 600, color: 'primary.main' }}>Market Value</TableCell>
                            <TableCell align="right" sx={{ fontWeight: 600, color: 'primary.main' }}>Unrealized P/L</TableCell>
                            <TableCell align="right" sx={{ fontWeight: 600, color: 'primary.main' }}>Current Price</TableCell>
                        </TableRow>
                    </TableHead>
                    <TableBody>
                        {positions.map((position) => (
                            <TableRow 
                                key={position.symbol}
                                sx={{ '&:hover': { bgcolor: 'rgba(33, 150, 243, 0.08)' } }}
                            >
                                <TableCell>{position.symbol}</TableCell>
                                <TableCell align="right">{position.qty}</TableCell>
                                <TableCell align="right">
                                    ${position.market_value.toLocaleString()}
                                </TableCell>
                                <TableCell align="right">
                                    ${position.unrealized_pl.toLocaleString()}
                                </TableCell>
                                <TableCell align="right">
                                    ${position.current_price.toLocaleString()}
                                </TableCell>
                            </TableRow>
                        ))}
                    </TableBody>
                </Table>
            </TableContainer>
        </Paper>
    );
};

export default PositionsTable; 