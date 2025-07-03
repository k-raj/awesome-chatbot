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

    // 6. Document Chunks Collection (for semantic chunking)
    createCollectionIfNotExists('document_chunks', {
        validator: {
            $jsonSchema: {
                bsonType: 'object',
                required: ['_id', 'document_id', 'content', 'created_at'],
                properties: {
                    _id: { bsonType: 'string' },
                    document_id: { bsonType: 'string' },
                    content: { bsonType: 'string' },
                    chunk_type: { 
                        bsonType: 'string',
                        enum: ['paragraph', 'heading', 'list', 'table', 'code', 'other']
                    },
                    chunk_index: { bsonType: 'int' },
                    start_idx: { bsonType: 'int' },
                    end_idx: { bsonType: 'int' },
                    token_count: { bsonType: 'int' },
                    embedding_model: { bsonType: 'string' },
                    metadata: { bsonType: 'object' },
                    created_at: { bsonType: 'date' }
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

    // =============================================================================
    // ADVANCED RAG SYSTEM COLLECTIONS
    // =============================================================================

    // 8. Retrieval Logs Collection (for performance monitoring)
    createCollectionIfNotExists('retrieval_logs', {
        validator: {
            $jsonSchema: {
                bsonType: 'object',
                required: ['_id', 'query', 'num_results', 'created_at'],
                properties: {
                    _id: { bsonType: 'string' },
                    session_id: { bsonType: 'string' },
                    query: { bsonType: 'string' },
                    query_embedding: { bsonType: 'array' },
                    num_results: { bsonType: 'int' },
                    retrieval_method: { 
                        bsonType: 'string',
                        enum: ['vector', 'bm25', 'hybrid']
                    },
                    vector_weight: { bsonType: 'double' },
                    bm25_weight: { bsonType: 'double' },
                    response_time_ms: { bsonType: 'int' },
                    avg_initial_score: { bsonType: 'double' },
                    avg_rerank_score: { bsonType: 'double' },
                    avg_final_score: { bsonType: 'double' },
                    chunk_types: { bsonType: 'array' },
                    filters: { bsonType: 'object' },
                    reranker_model: { bsonType: 'string' },
                    embedding_model: { bsonType: 'string' },
                    created_at: { bsonType: 'date' }
                }
            }
        }
    });

    // 9. Performance Metrics Collection
    createCollectionIfNotExists('performance_metrics', {
        validator: {
            $jsonSchema: {
                bsonType: 'object',
                required: ['_id', 'metric_type', 'value', 'timestamp'],
                properties: {
                    _id: { bsonType: 'string' },
                    metric_type: { 
                        bsonType: 'string',
                        enum: ['latency', 'throughput', 'accuracy', 'relevance', 'cache_hit_rate', 'token_usage']
                    },
                    value: { bsonType: 'double' },
                    unit: { bsonType: 'string' },
                    service: { bsonType: 'string' },
                    session_id: { bsonType: 'string' },
                    group_id: { bsonType: 'string' },
                    metadata: { bsonType: 'object' },
                    timestamp: { bsonType: 'date' }
                }
            }
        }
    });

    // 10. Model Configurations Collection
    createCollectionIfNotExists('model_configurations', {
        validator: {
            $jsonSchema: {
                bsonType: 'object',
                required: ['_id', 'model_type', 'model_name', 'created_at'],
                properties: {
                    _id: { bsonType: 'string' },
                    model_type: { 
                        bsonType: 'string',
                        enum: ['embedding', 'llm', 'reranker']
                    },
                    model_name: { bsonType: 'string' },
                    model_version: { bsonType: 'string' },
                    provider: { bsonType: 'string' },
                    configuration: { bsonType: 'object' },
                    is_active: { bsonType: 'bool' },
                    performance_stats: { bsonType: 'object' },
                    created_at: { bsonType: 'date' },
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

    // 12. System Logs Collection (for debugging and monitoring)
    createCollectionIfNotExists('system_logs', {
        validator: {
            $jsonSchema: {
                bsonType: 'object',
                required: ['_id', 'level', 'message', 'timestamp'],
                properties: {
                    _id: { bsonType: 'string' },
                    level: { 
                        bsonType: 'string',
                        enum: ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
                    },
                    message: { bsonType: 'string' },
                    service: { bsonType: 'string' },
                    component: { bsonType: 'string' },
                    session_id: { bsonType: 'string' },
                    user_id: { bsonType: 'string' },
                    error_code: { bsonType: 'string' },
                    stack_trace: { bsonType: 'string' },
                    metadata: { bsonType: 'object' },
                    timestamp: { bsonType: 'date' }
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

    // Document Chunks Indexes
    createIndexIfNotExists(db.document_chunks, { 'document_id': 1, 'chunk_index': 1 });
    createIndexIfNotExists(db.document_chunks, { 'chunk_type': 1 });
    createIndexIfNotExists(db.document_chunks, { 'content': 'text' });
    createIndexIfNotExists(db.document_chunks, { 'token_count': 1 });
    createIndexIfNotExists(db.document_chunks, { 'embedding_model': 1 });
    createIndexIfNotExists(db.document_chunks, { 'created_at': -1 });

    // File Uploads Indexes
    createIndexIfNotExists(db.file_uploads, { 'filename': 1 });
    createIndexIfNotExists(db.file_uploads, { 'file_type': 1 });
    createIndexIfNotExists(db.file_uploads, { 'upload_status': 1 });
    createIndexIfNotExists(db.file_uploads, { 'group_id': 1 });
    createIndexIfNotExists(db.file_uploads, { 'uploaded_by': 1 });
    createIndexIfNotExists(db.file_uploads, { 'checksum': 1 });
    createIndexIfNotExists(db.file_uploads, { 'created_at': -1 });
    createIndexIfNotExists(db.file_uploads, { 'processed_at': -1 });

    // Retrieval Logs Indexes
    createIndexIfNotExists(db.retrieval_logs, { 'session_id': 1 });
    createIndexIfNotExists(db.retrieval_logs, { 'query': 'text' });
    createIndexIfNotExists(db.retrieval_logs, { 'retrieval_method': 1 });
    createIndexIfNotExists(db.retrieval_logs, { 'response_time_ms': 1 });
    createIndexIfNotExists(db.retrieval_logs, { 'created_at': -1 });

    // Performance Metrics Indexes
    createIndexIfNotExists(db.performance_metrics, { 'metric_type': 1, 'timestamp': -1 });
    createIndexIfNotExists(db.performance_metrics, { 'service': 1, 'timestamp': -1 });
    createIndexIfNotExists(db.performance_metrics, { 'session_id': 1 });
    createIndexIfNotExists(db.performance_metrics, { 'group_id': 1 });

    // Model Configurations Indexes
    createIndexIfNotExists(db.model_configurations, { 'model_type': 1 });
    createIndexIfNotExists(db.model_configurations, { 'model_name': 1 });
    createIndexIfNotExists(db.model_configurations, { 'is_active': 1 });
    createIndexIfNotExists(db.model_configurations, { 'provider': 1 });

    // User Sessions Indexes
    createIndexIfNotExists(db.user_sessions, { 'user_id': 1 });
    createIndexIfNotExists(db.user_sessions, { 'session_token': 1 }, { unique: true });
    createIndexIfNotExists(db.user_sessions, { 'is_active': 1 });
    createIndexIfNotExists(db.user_sessions, { 'last_activity': -1 });
    createIndexIfNotExists(db.user_sessions, { 'expires_at': 1 });

    // System Logs Indexes
    createIndexIfNotExists(db.system_logs, { 'level': 1, 'timestamp': -1 });
    createIndexIfNotExists(db.system_logs, { 'service': 1, 'timestamp': -1 });
    createIndexIfNotExists(db.system_logs, { 'component': 1 });
    createIndexIfNotExists(db.system_logs, { 'session_id': 1 });
    createIndexIfNotExists(db.system_logs, { 'user_id': 1 });
    createIndexIfNotExists(db.system_logs, { 'error_code': 1 });


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

    // Insert default model configurations
    try {
        if (db.model_configurations.countDocuments() === 0) {
            const defaultModels = [
                {
                    _id: 'embedding-default',
                    model_type: 'embedding',
                    model_name: process.env.EMBEDDING_MODEL || 'all-MiniLM-L6-v2',
                    model_version: 'latest',
                    provider: 'sentence-transformers',
                    configuration: {
                        max_seq_length: 512,
                        normalize_embeddings: true
                    },
                    is_active: true,
                    performance_stats: {},
                    created_at: new Date(),
                    updated_at: new Date()
                },
                {
                    _id: 'reranker-default',
                    model_type: 'reranker',
                    model_name: process.env.RERANKER_MODEL || 'cross-encoder/ms-marco-MiniLM-L-6-v2',
                    model_version: 'latest',
                    provider: 'sentence-transformers',
                    configuration: {
                        max_length: 512
                    },
                    is_active: true,
                    performance_stats: {},
                    created_at: new Date(),
                    updated_at: new Date()
                }
            ];
            
            db.model_configurations.insertMany(defaultModels);
            print('Created default model configurations');
        }
    } catch (error) {
        print(`Default model configurations creation handled: ${error.message}`);
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