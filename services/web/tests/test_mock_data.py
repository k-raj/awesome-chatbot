"""
Mock data generators for comprehensive testing
"""

import random
from datetime import datetime, timedelta
from faker import Faker

fake = Faker()


class MockDataGenerator:
    """Generate realistic mock data for testing"""
    
    @staticmethod
    def generate_groups(count=10):
        """Generate mock groups data"""
        groups = []
        for i in range(count):
            created_at = fake.date_time_between(start_date='-30d', end_date='now')
            groups.append({
                '_id': f'group_{i}',
                'name': fake.catch_phrase(),
                'description': fake.text(max_nb_chars=200),
                'file_count': random.randint(0, 50),
                'message_count': random.randint(0, 200),
                'created_at': created_at.isoformat(),
                'updated_at': (created_at + timedelta(days=random.randint(0, 10))).isoformat()
            })
        
        # Always include General Chat
        groups.insert(0, {
            '_id': 'general',
            'name': 'General Chat',
            'description': 'General conversation without document context',
            'file_count': 0,
            'message_count': random.randint(10, 100),
            'created_at': (datetime.now() - timedelta(days=30)).isoformat(),
            'is_default': True
        })
        
        return {'groups': groups}
    
    @staticmethod
    def generate_files(count=20):
        """Generate mock files data"""
        files = []
        statuses = ['completed', 'processing', 'failed', 'skipped']
        file_types = ['pdf', 'txt', 'doc', 'docx']
        
        for i in range(count):
            file_type = random.choice(file_types)
            status = random.choice(statuses)
            
            files.append({
                '_id': f'file_{i}',
                'filename': f'{fake.file_name(extension=file_type)}',
                'original_filename': f'{fake.file_name(extension=file_type)}',
                'file_size': random.randint(1024, 10 * 1024 * 1024),  # 1KB to 10MB
                'file_type': file_type,
                'upload_status': status,
                'processing_progress': 100 if status == 'completed' else random.randint(0, 99),
                'chunks_count': random.randint(5, 100) if status == 'completed' else 0,
                'created_at': fake.date_time_between(start_date='-7d', end_date='now').isoformat(),
                'processed_at': fake.date_time_between(start_date='-6d', end_date='now').isoformat() if status == 'completed' else None
            })
        
        return {'files': files, 'total_files': count}
    
    @staticmethod
    def generate_sessions(count=15):
        """Generate mock chat sessions"""
        sessions = []
        
        for i in range(count):
            last_message = fake.date_time_between(start_date='-7d', end_date='now')
            sessions.append({
                'session_id': f'session_{i}',
                'title': fake.sentence(nb_words=6).rstrip('.'),
                'message_count': random.randint(2, 50),
                'last_message_at': last_message.isoformat(),
                'created_at': (last_message - timedelta(hours=random.randint(1, 48))).isoformat(),
                'preview': fake.text(max_nb_chars=100),
                'status': 'active'
            })
        
        # Sort by last_message_at descending
        sessions.sort(key=lambda x: x['last_message_at'], reverse=True)
        
        return {
            'sessions': sessions,
            'pagination': {
                'page': 1,
                'limit': 20,
                'total': count,
                'pages': 1
            }
        }
    
    @staticmethod
    def generate_messages(count=20):
        """Generate mock chat messages"""
        messages = []
        current_time = datetime.now()
        
        for i in range(count):
            message_time = current_time - timedelta(minutes=count - i)
            is_user = i % 2 == 0
            
            message = {
                '_id': f'msg_{i}',
                'message_type': 'user' if is_user else 'assistant',
                'content': fake.paragraph(nb_sentences=random.randint(1, 4)),
                'created_at': message_time.isoformat()
            }
            
            # Add context for assistant messages
            if not is_user and random.choice([True, False]):
                message['context_used'] = [
                    {
                        'content': fake.paragraph(nb_sentences=2),
                        'metadata': {
                            'filename': fake.file_name(extension='pdf'),
                            'page': random.randint(1, 20)
                        }
                    }
                    for _ in range(random.randint(1, 3))
                ]
                message['model_used'] = 'llama3.2:3b'
                message['generation_time'] = random.uniform(0.5, 3.0)
            
            messages.append(message)
        
        return messages
    
    @staticmethod
    def generate_chat_response():
        """Generate mock chat API response"""
        return {
            'session_id': f'session_{random.randint(1000, 9999)}',
            'message_id': f'msg_{random.randint(1000, 9999)}',
            'response': fake.paragraph(nb_sentences=random.randint(2, 5)),
            'retrieved_context': [
                {
                    'id': f'chunk_{i}',
                    'content': fake.paragraph(nb_sentences=3),
                    'metadata': {
                        'filename': fake.file_name(extension='pdf'),
                        'document_id': f'doc_{i}',
                        'chunk_type': 'paragraph'
                    },
                    'relevance_score': random.uniform(0.6, 0.95)
                }
                for i in range(random.randint(0, 3))
            ],
            'task_id': f'task_{random.randint(1000, 9999)}',
            'group_type': random.choice(['general', 'rag'])
        }
