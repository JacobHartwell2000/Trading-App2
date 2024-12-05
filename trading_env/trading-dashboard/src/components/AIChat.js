import React, { useState, useRef, useEffect } from 'react';
import {
    Box,
    TextField,
    IconButton,
    List,
    ListItem,
    ListItemText,
    Paper,
    Typography,
    Divider
} from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import axios from 'axios';

function AIChat() {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');
    const messagesEndRef = useRef(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const handleSend = async () => {
        if (!input.trim()) return;

        // Add user message
        setMessages(prev => [...prev, { text: input, sender: 'user' }]);
        
        try {
            // Send message to backend
            const response = await axios.post('http://localhost:5000/api/chat', {
                message: input
            });

            // Add bot response with context
            if (response.data.status === 'success') {
                setMessages(prev => [...prev, {
                    text: response.data.data.response,
                    sender: 'bot',
                    context: response.data.data.context
                }]);
            }
        } catch (error) {
            console.error('Error sending message:', error);
            setMessages(prev => [...prev, {
                text: 'Sorry, there was an error processing your request.',
                sender: 'bot'
            }]);
        }

        setInput('');
    };

    const handleKeyPress = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };

    return (
        <Paper elevation={3} sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            <Box sx={{ p: 2, flexGrow: 1, overflow: 'auto' }}>
                <List>
                    {messages.map((message, index) => (
                        <React.Fragment key={index}>
                            <ListItem alignItems="flex-start">
                                <ListItemText
                                    primary={message.text}
                                    secondary={message.context && (
                                        <Typography
                                            component="div"
                                            variant="body2"
                                            sx={{
                                                mt: 1,
                                                color: 'text.secondary',
                                                fontSize: '0.85rem',
                                                fontStyle: 'italic',
                                                bgcolor: 'rgba(0,0,0,0.1)',
                                                p: 1,
                                                borderRadius: 1
                                            }}
                                        >
                                            Reasoning: {message.context}
                                        </Typography>
                                    )}
                                    sx={{
                                        '& .MuiListItemText-primary': {
                                            bgcolor: message.sender === 'user' ? 'primary.main' : 'background.paper',
                                            p: 2,
                                            borderRadius: 2,
                                            display: 'inline-block',
                                            maxWidth: '80%'
                                        }
                                    }}
                                />
                            </ListItem>
                            {index < messages.length - 1 && <Divider />}
                        </React.Fragment>
                    ))}
                    <div ref={messagesEndRef} />
                </List>
            </Box>
            <Box sx={{ display: 'flex', gap: 1 }}>
                <TextField
                    fullWidth
                    variant="outlined"
                    placeholder="Ask about trading strategies, market analysis, or current positions..."
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyPress={handleKeyPress}
                    sx={{ bgcolor: 'background.paper' }}
                />
                <IconButton
                    color="primary"
                    onClick={handleSend}
                    sx={{ bgcolor: 'background.paper' }}
                >
                    <SendIcon />
                </IconButton>
            </Box>
        </Paper>
    );
}

export default AIChat; 