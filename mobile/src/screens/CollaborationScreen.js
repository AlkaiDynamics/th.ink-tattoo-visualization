import React, { useState, useEffect, useCallback } from 'react';
import { View, StyleSheet, Alert } from 'react-native';
import { GiftedChat } from 'react-native-gifted-chat';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { firebase } from '@react-native-firebase/app';
import '@react-native-firebase/auth';
import '@react-native-firebase/firestore';

// Initialize Firebase if not already initialized
const firebaseConfig = {
  apiKey: "YOUR_FIREBASE_API_KEY",
  authDomain: "YOUR_FIREBASE_AUTH_DOMAIN",
  projectId: "YOUR_FIREBASE_PROJECT_ID",
  storageBucket: "YOUR_FIREBASE_STORAGE_BUCKET",
  messagingSenderId: "YOUR_FIREBASE_SENDER_ID",
  appId: "YOUR_FIREBASE_APP_ID"
};

if (!firebase.apps.length) {
  firebase.initializeApp(firebaseConfig);
}

const CollaborationScreen = () => {
  const [messages, setMessages] = useState([]);
  const [user, setUser] = useState(null);

  const fetchUser = async () => {
    try {
      const token = await AsyncStorage.getItem('token');
      // Decode the JWT to get user information (this is a simplified example)
      const payload = JSON.parse(atob(token.split('.')[1]));
      setUser({
        _id: payload.sub,
        name: payload.sub,
        avatar: 'https://placeimg.com/140/140/any', // Placeholder avatar
      });
    } catch (error) {
      Alert.alert('Error', 'Failed to fetch user information.');
      console.error(error);
    }
  };

  useEffect(() => {
    fetchUser();
  }, []);

  useEffect(() => {
    if (user) {
      const unsubscribe = firebase.firestore()
        .collection('chats')
        .orderBy('createdAt', 'desc')
        .onSnapshot(querySnapshot => {
          const messagesFirestore = querySnapshot.docs.map(doc => {
            const data = doc.data();
            return {
              _id: data._id,
              text: data.text,
              createdAt: data.createdAt.toDate(),
              user: {
                _id: data.user._id,
                name: data.user.name,
                avatar: data.user.avatar,
              },
            };
          });
          setMessages(messagesFirestore);
        }, error => {
          Alert.alert('Error', 'Failed to fetch messages.');
          console.error(error);
        });

      return () => unsubscribe();
    }
  }, [user]);

  const onSend = useCallback((messages = []) => {
    const { _id, text, createdAt, user: messageUser } = messages[0];
    firebase.firestore().collection('chats').add({
      _id,
      text,
      createdAt,
      user: messageUser,
    }).catch(error => {
      Alert.alert('Error', 'Failed to send message.');
      console.error(error);
    });
  }, []);

  if (!user) {
    return <View style={styles.loaderContainer}><ActivityIndicator size="large" color="#0000ff" /></View>;
  }

  return (
    <GiftedChat
      messages={messages}
      onSend={messages => onSend(messages)}
      user={user}
      placeholder="Type a message..."
      showAvatarForEveryMessage
      showUserAvatar
    />
  );
};

export default CollaborationScreen;

const styles = StyleSheet.create({
  loaderContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
});