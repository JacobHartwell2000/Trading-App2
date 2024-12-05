import React from 'react';
import { Container, CssBaseline, AppBar, Toolbar, Typography, ThemeProvider } from '@mui/material';
import Dashboard from './components/Dashboard';
import { theme } from './theme';

function App() {
    return (
        <ThemeProvider theme={theme}>
            <CssBaseline />
            <AppBar position="static" elevation={0} sx={{ bgcolor: 'background.paper', borderBottom: 1, borderColor: 'divider' }}>
                <Toolbar>
                    <Typography variant="h6" sx={{ color: 'primary.main', fontWeight: 600 }}>
                        AI Trading Dashboard
                    </Typography>
                </Toolbar>
            </AppBar>
            <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
                <Dashboard />
            </Container>
        </ThemeProvider>
    );
}

export default App;
