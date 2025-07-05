// init-mongo.js - MongoDB initialization script for Advanced RAG System
// Based on README.md specifications and environment configuration

try {
    // Switch to the admin database first to create admin user
    db = db.getSiblingDB('admin');
    print('Connected to admin database');

    // Create admin user if it doesn't exist (from MONGO_INITDB_ROOT_USERNAME/PASSWORD)
    const adminUser =  process.env.MONGO_INITDB_ROOT_USERNAME;
    const adminPassword =process.env.MONGO_INITDB_ROOT_PASSWORD;
    const databaseName = process.env.MONGO_INITDB_DATABASE || 'ragdb'
    try {
        const existingAdmin = db.getUser(adminUser);
        if (!existingAdmin) {
            db.createUser({
                user: adminUser,
                pwd: adminPassword,
                roles: [
                    { role: 'userAdminAnyDatabase', db: 'admin' },
                    { role: 'readWriteAnyDatabase', db: 'admin' },
                    { role: 'dbAdminAnyDatabase', db: 'admin' },
                    { role: 'clusterAdmin', db: 'admin' }
                ]
            });
            print(`Created admin user: ${adminUser}`);
        } else {
            print(`Admin user ${adminUser} already exists`);
        }
    } catch (error) {
        print(`Admin user creation handled: ${error.message}`);
    }

    // Switch to the RAG database (from MONGO_INITDB_DATABASE)
    db = db.getSiblingDB(databaseName);
    print('Connected to RAG database:', databaseName);

    // Create application user for RAG system
    const ragUser = adminUser;
    const ragPassword = adminPassword;
    
    try {
        const existingUser = db.getUser(ragUser);
        if (!existingUser) {
            db.createUser({
                user: ragUser,
                pwd: ragPassword,
                roles: [
                    {
                        role: 'readWrite',
                        db: databaseName
                    },
                    {
                        role: 'dbAdmin',
                        db: databaseName
                    }
                ]
            });
            print(`Created RAG user: ${ragUser}`);
        } else {
            print(`RAG user ${ragUser} already exists`);
        }
    } catch (error) {
        print(`RAG user creation handled: ${error.message}`);
    }

    // Function to create collection if it doesn't exist
    function createCollectionIfNotExists(collectionName, options = {}) {
        const collections = db.getCollectionNames();
        if (!collections.includes(collectionName)) {
            db.createCollection(collectionName, options);
            print(`Created collection: ${collectionName}`);
        } else {
            print(`Collection ${collectionName} already exists`);
        }
    }

    // Function to create index if it doesn't exist
    function createIndexIfNotExists(collection, indexSpec, options = {}) {
        try {
            collection.createIndex(indexSpec, options);
            print(`Created index on ${collection.getName()}: ${JSON.stringify(indexSpec)}`);
        } catch (error) {
            if (error.code === 85) { // Index already exists
                print(`Index already exists on ${collection.getName()}: ${JSON.stringify(indexSpec)}`);
            } else {
                print(`Error creating index on ${collection.getName()}: ${error.message}`);
            }
        }
    }

    // =============================================================================
    // CORE RAG SYSTEM COLLECTIONS
    // =============================================================================

    // 1. Content Groups Collection (for organizing chats by topics/projects)
    createCollectionIfNotExists('content_groups', {
        validator: {
            $jsonSchema: {
                bsonType: 'object',
                required: ['_id', 'name', 'created_at'],
                properties: {
                    _id: { bsonType: 'string' },
                    name: { bsonType: 'string' },
                    description: { bsonType: 'string' },
                    created_by: { bsonType: 'string' },
                    settings: { bsonType: 'object' },
                    created_at: { bsonType: 'date' },
                    updated_at: { bsonType: 'date' }
                }
            }
        }
    });

    // 2. Chat Sessions Collection (for multi-turn conversations)
    createCollectionIfNotExists('chat_sessions', {
        validator: {
            $jsonSchema: {
                bsonType: 'object',
                required: ['_id', 'user_id', 'group_id', 'created_at'],
                properties: {
                    _id: { bsonType: 'string' },
                    user_id: { bsonType: 'string' },
                    group_id: { bsonType: 'string' },
                    title: { bsonType: 'string' },
                    status: { 
                        bsonType: 'string',
                        enum: ['active', 'archived', 'deleted']
                    },
                    last_message_at: { bsonType: 'date' },
                    message_count: { bsonType: 'int' },
                    created_at: { bsonType: 'date' },
                    updated_at: { bsonType: 'date' }
                }
            }
        }
    });

    // 3. Messages Collection (enhanced for multi-turn conversations)
    createCollectionIfNotExists('messages', {
        validator: {
            $jsonSchema: {
                bsonType: 'object',
                required: ['_id', 'session_id', 'message_type', 'content', 'created_at'],
                properties: {
                    _id: { bsonType: 'string' },
                    session_id: { bsonType: 'string' },
                    user_id: { bsonType: 'string' },
                    group_id: { bsonType: 'string' },
                    message_type: { 
                        bsonType: 'string',
                        enum: ['user', 'assistant', 'system']
                    },
                    content: { bsonType: 'string' },
                    query: { bsonType: 'string' },
                    response: { bsonType: 'string' },
                    context_used: { bsonType: 'array' },
                    retrieval_metadata: { bsonType: 'object' },
                    query_embedding: { bsonType: 'array' },
                    response_embedding: { bsonType: 'array' },
                    token_count: { bsonType: 'int' },
                    processing_time_ms: { bsonType: 'int' },
                    model_used: { bsonType: 'string' },
                    created_at: { bsonType: 'date' },
                    embeddings_computed_at: { bsonType: 'date' }
                }
            }
        }
    });

    // 4. User Feedback Collection (enhanced with detailed feedback)
    createCollectionIfNotExists('user_feedback', {
        validator: {
            $jsonSchema: {
                bsonType: 'object',
                required: ['_id', 'message_id', 'feedback_type', 'created_at'],
                properties: {
                    _id: { bsonType: 'string' },
                    message_id: { bsonType: 'string' },
                    session_id: { bsonType: 'string' },
                    user_id: { bsonType: 'string' },
                    feedback_type: { 
                        bsonType: 'string',
                        enum: ['thumbs_up', 'thumbs_down', 'flag', 'suggestion']
                    },
                    rating: { bsonType: 'int' }, // 1-5 scale
                    comment: { bsonType: 'string' },
                    categories: { bsonType: 'array' }, // helpfulness, accuracy, relevance, etc.
                    metadata: { bsonType: 'object' },
                    created_at: { bsonType: 'date' }
                }
            }
        }
    });

    // =============================================================================
    // DOCUMENT MANAGEMENT COLLECTIONS
    // =============================================================================

    // 5. Indexed Documents Collection (for file upload support)
    createCollectionIfNotExists('indexed_documents', {
        validator: {
            $jsonSchema: {
                bsonType: 'object',
                required: ['_id', 'content', 'metadata', 'created_at'],
                properties: {
                    _id: { bsonType: 'string' },
                    content: { bsonType: 'string' },
                    title: { bsonType: 'string' },
                    source: { bsonType: 'string' },
                    file_path: { bsonType: 'string' },
                    file_type: { bsonType: 'string' },
                    file_size: { bsonType: 'int' },
                    chunk_count: { bsonType: 'int' },
                    group_id: { bsonType: 'string' },
                    uploaded_by: { bsonType: 'string' },
                    processing_status: { 
                        bsonType: 'string',
                        enum: ['pending', 'processing', 'completed', 'failed']
                    },
                    metadata: { bsonType: 'object' },
                    indexed_at: { bsonType: 'date' },
                    created_at: { bsonType: 'date' },
                    updated_at: { bsonType: 'date' }
                }
            }
        }
    });


    // 7. File Uploads Collection (enhanced)
    createCollectionIfNotExists('file_uploads', {
        validator: {
            $jsonSchema: {
                bsonType: 'object',
                required: ['_id', 'filename', 'file_type', 'upload_status', 'created_at'],
                properties: {
                    _id: { bsonType: 'string' },
                    filename: { bsonType: 'string' },
                    original_filename: { bsonType: 'string' },
                    file_path: { bsonType: 'string' },
                    file_type: { bsonType: 'string' },
                    file_size: { bsonType: 'int' },
                    mime_type: { bsonType: 'string' },
                    group_id: { bsonType: 'string' },
                    uploaded_by: { bsonType: 'string' },
                    upload_status: { 
                        bsonType: 'string',
                        enum: ['pending', 'processing', 'completed', 'failed']
                    },
                    processing_progress: { bsonType: 'int' }, // 0-100
                    chunks_count: { bsonType: 'int' },
                    error_message: { bsonType: 'string' },
                    checksum: { bsonType: 'string' },
                    metadata: { bsonType: 'object' },
                    created_at: { bsonType: 'date' },
                    processed_at: { bsonType: 'date' },
                    updated_at: { bsonType: 'date' }
                }
            }
        }
    });


    // 11. User Sessions Collection (for authentication and tracking)
    createCollectionIfNotExists('user_sessions', {
        validator: {
            $jsonSchema: {
                bsonType: 'object',
                required: ['_id', 'user_id', 'created_at'],
                properties: {
                    _id: { bsonType: 'string' },
                    user_id: { bsonType: 'string' },
                    session_token: { bsonType: 'string' },
                    ip_address: { bsonType: 'string' },
                    user_agent: { bsonType: 'string' },
                    is_active: { bsonType: 'bool' },
                    last_activity: { bsonType: 'date' },
                    created_at: { bsonType: 'date' },
                    expires_at: { bsonType: 'date' }
                }
            }
        }
    });


    // 13. Users Collection (for user management and group access)
    createCollectionIfNotExists('users', {
        validator: {
            $jsonSchema: {
                bsonType: 'object',
                required: ['_id', 'username', 'email', 'created_at'],
                properties: {
                    _id: { bsonType: 'string' },
                    username: { bsonType: 'string' },
                    email: { bsonType: 'string' },
                    password_hash: { bsonType: 'string' },
                    first_name: { bsonType: 'string' },
                    last_name: { bsonType: 'string' },
                    is_active: { bsonType: 'bool' },
                    is_admin: { bsonType: 'bool' },
                    group_ids: { 
                        bsonType: 'array',
                        items: { bsonType: 'string' }
                    },
                    default_group_id: { bsonType: 'string' },
                    preferences: { 
                        bsonType: 'object',
                        properties: {
                            theme: { bsonType: 'string' },
                            language: { bsonType: 'string' },
                            notifications: { bsonType: 'bool' },
                            model_preference: { bsonType: 'string' },
                            max_context_length: { bsonType: 'int' }
                        }
                    },
                    api_key: { bsonType: 'string' },
                    last_login: { bsonType: 'date' },
                    login_count: { bsonType: 'int' },
                    profile_picture: { bsonType: 'string' },
                    metadata: { bsonType: 'object' },
                    created_at: { bsonType: 'date' },
                    updated_at: { bsonType: 'date' }
                }
            }
        }
    });

    // =============================================================================
    // CREATE INDEXES FOR PERFORMANCE OPTIMIZATION
    // =============================================================================

    print('Creating indexes for performance optimization...');

    // Content Groups Indexes
    createIndexIfNotExists(db.content_groups, { 'name': 1 }, { unique: true });
    createIndexIfNotExists(db.content_groups, { 'created_by': 1 });
    createIndexIfNotExists(db.content_groups, { 'created_at': -1 });

    // Chat Sessions Indexes
    createIndexIfNotExists(db.chat_sessions, { 'user_id': 1, 'group_id': 1 });
    createIndexIfNotExists(db.chat_sessions, { 'user_id': 1, 'last_message_at': -1 });
    createIndexIfNotExists(db.chat_sessions, { 'status': 1 });
    createIndexIfNotExists(db.chat_sessions, { 'created_at': -1 });

    // Messages Indexes
    createIndexIfNotExists(db.messages, { 'session_id': 1, 'created_at': -1 });
    createIndexIfNotExists(db.messages, { 'user_id': 1, 'created_at': -1 });
    createIndexIfNotExists(db.messages, { 'group_id': 1, 'created_at': -1 });
    createIndexIfNotExists(db.messages, { 'message_type': 1 });
    createIndexIfNotExists(db.messages, { 'content': 'text' });

    // User Feedback Indexes
    createIndexIfNotExists(db.user_feedback, { 'message_id': 1 });
    createIndexIfNotExists(db.user_feedback, { 'session_id': 1 });
    createIndexIfNotExists(db.user_feedback, { 'user_id': 1, 'created_at': -1 });
    createIndexIfNotExists(db.user_feedback, { 'feedback_type': 1 });
    createIndexIfNotExists(db.user_feedback, { 'rating': 1 });

    // Indexed Documents Indexes
    createIndexIfNotExists(db.indexed_documents, { 'title': 'text', 'content': 'text' });
    createIndexIfNotExists(db.indexed_documents, { 'source': 1 });
    createIndexIfNotExists(db.indexed_documents, { 'file_type': 1 });
    createIndexIfNotExists(db.indexed_documents, { 'group_id': 1 });
    createIndexIfNotExists(db.indexed_documents, { 'uploaded_by': 1 });
    createIndexIfNotExists(db.indexed_documents, { 'processing_status': 1 });
    createIndexIfNotExists(db.indexed_documents, { 'indexed_at': -1 });
    createIndexIfNotExists(db.indexed_documents, { 'created_at': -1 });


    // File Uploads Indexes
    createIndexIfNotExists(db.file_uploads, { 'filename': 1 });
    createIndexIfNotExists(db.file_uploads, { 'file_type': 1 });
    createIndexIfNotExists(db.file_uploads, { 'upload_status': 1 });
    createIndexIfNotExists(db.file_uploads, { 'group_id': 1 });
    createIndexIfNotExists(db.file_uploads, { 'uploaded_by': 1 });
    createIndexIfNotExists(db.file_uploads, { 'checksum': 1 });
    createIndexIfNotExists(db.file_uploads, { 'created_at': -1 });
    createIndexIfNotExists(db.file_uploads, { 'processed_at': -1 });


    // User Sessions Indexes
    createIndexIfNotExists(db.user_sessions, { 'user_id': 1 });
    createIndexIfNotExists(db.user_sessions, { 'session_token': 1 }, { unique: true });
    createIndexIfNotExists(db.user_sessions, { 'is_active': 1 });
    createIndexIfNotExists(db.user_sessions, { 'last_activity': -1 });
    createIndexIfNotExists(db.user_sessions, { 'expires_at': 1 });


    // Users Indexes
    createIndexIfNotExists(db.users, { 'username': 1 }, { unique: true });
    createIndexIfNotExists(db.users, { 'email': 1 }, { unique: true });
    createIndexIfNotExists(db.users, { 'api_key': 1 }, { unique: true, sparse: true });
    createIndexIfNotExists(db.users, { 'group_ids': 1 });
    createIndexIfNotExists(db.users, { 'is_active': 1 });
    createIndexIfNotExists(db.users, { 'last_login': -1 });
    createIndexIfNotExists(db.users, { 'created_at': -1 });


    // =============================================================================
    // INSERT INITIAL DATA
    // =============================================================================


    // Insert default content group
    try {
        if (db.content_groups.countDocuments() === 0) {
            db.content_groups.insertOne({
                _id: 'default',
                name: 'General',
                description: 'Default content group for general conversations',
                created_by: 'system',
                settings: {
                    max_history: 50,
                    auto_archive: false,
                    embedding_model:  process.env.EMBEDDING_MODEL || 'all-MiniLM-L6-v2',
                },
                created_at: new Date(),
                updated_at: new Date()
            });
            print('Created default content group');
        }
    } catch (error) {
        print(`Default content group creation handled: ${error.message}`);
    }


    // Create default admin user
    try {
        if (db.users.countDocuments() === 0) {
            const defaultAdminId = 'user-007';
            db.users.insertOne({
                _id: 'user007-uid',
                username: 'user007',
                email: 'user1@ragchatbot.local',
                password_hash: '$2b$12$dummy.hash.for.initial.setup', // Should be changed on first login
                first_name: 'user',
                last_name: '007',
                is_active: true,
                is_admin: true,
                group_ids: ['default'],
                default_group_id: 'default',
                preferences: {
                    theme: 'dark',
                    language: 'en',
                    notifications: true,
                    model_preference: 'auto',
                    max_context_length: 2048
                },
                login_count: 0,
                metadata: {
                    created_by: 'system',
                    account_type: 'system_admin'
                },
                created_at: new Date(),
                updated_at: new Date()
            });
            print('Created default user ');
        }
    } catch (error) {
        print(`Default admin user creation handled: ${error.message}`);
    }

    // =============================================================================
    // FINAL VERIFICATION
    // =============================================================================

    print('=============================================================================');
    print('MONGODB INITIALIZATION COMPLETED SUCCESSFULLY');
    print('=============================================================================');
    
    // Verify collections
    const collections = db.getCollectionNames();
    print(`Total collections created: ${collections.length}`);
    print('Collections:');
    collections.forEach(collection => {
        const count = db.getCollection(collection).countDocuments();
        print(`  - ${collection}: ${count} documents`);
    });

    // Verify users
    const users = db.getUsers();
    print(`\nTotal users: ${users.length}`);
    if(users.length > 0) {
        users.forEach(user => {
            print(`  - ${user.user}: ${user.roles.map(r => r.role).join(', ')}`);
        });
    }

    print('\nRAG System MongoDB setup completed successfully!');
    print('Ready for production use with all collections and indexes configured.');

} catch (error) {
    print('=============================================================================');
    print('ERROR DURING MONGODB INITIALIZATION');
    print('=============================================================================');
    print('Error: ' + error.message);
    print('Stack: ' + error.stack);
    throw error;
}