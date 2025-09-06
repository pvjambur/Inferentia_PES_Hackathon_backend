from fastapi import APIRouter, HTTPException, Depends, Form
from typing import List, Optional, Dict, Any
import json
import numpy as np
import logging
import os
from database.json_db import JSONDatabase
from core.langraph.coordinator import LangGraphCoordinator
from core.training.ml_trainer import MLTrainer
from utils.data_processing import DataProcessor

logger = logging.getLogger(__name__)
router = APIRouter()

async def get_db() -> JSONDatabase:
    from main import app
    return app.state.db

@router.post("/generate/{model_id}")
async def generate_questionnaire(
    model_id: str,
    num_questions: int = Form(5),
    question_type: str = Form("yes_no"),  # yes_no, multiple_choice, open_ended
    sample_size: int = Form(10),
    db: JSONDatabase = Depends(get_db)
):
    """Generate questionnaire based on model predictions"""
    try:
        # Get model info
        models = await db.get_models()
        if model_id not in models:
            raise HTTPException(status_code=404, detail="Model not found")
            
        model_info = models[model_id]
        
        # Load model and make predictions on random samples
        trainer = MLTrainer()
        model_data = await trainer.load_model(model_info['file_path'])
        
        # Generate random test samples (in production, use actual test data)
        processor = DataProcessor()
        
        # For demo, create random samples based on model input shape
        # In production, this should come from actual test dataset
        if 'input_shape' in model_info:
            input_shape = model_info['input_shape']
        else:
            # Estimate input shape from model
            input_shape = 10  # Default
            
        X_samples = np.random.randn(sample_size, input_shape)
        predictions = await trainer.predict(model_info['file_path'], X_samples)
        
        # Generate questions based on predictions
        questions = []
        
        for i in range(num_questions):
            # Select random sample and prediction
            sample_idx = np.random.randint(0, len(predictions))
            prediction = predictions[sample_idx]
            sample_data = X_samples[sample_idx]
            
            if question_type == "yes_no":
                question = await _generate_yes_no_question(
                    sample_data, prediction, model_info, i
                )
            elif question_type == "multiple_choice":
                question = await _generate_multiple_choice_question(
                    sample_data, prediction, model_info, i
                )
            else:
                question = await _generate_open_ended_question(
                    sample_data, prediction, model_info, i
                )
            
            questions.append(question)
        
        # Save questionnaire
        questionnaire_data = {
            'model_id': model_id,
            'agent_id': model_info['agent_id'],
            'questions': questions,
            'question_type': question_type,
            'num_questions': num_questions,
            'status': 'generated'
        }
        
        questionnaire_id = await db.create_questionnaire(questionnaire_data)
        
        return {
            'questionnaire_id': questionnaire_id,
            'model_id': model_id,
            'questions': questions,
            'total_questions': len(questions),
            'question_type': question_type
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating questionnaire: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def _generate_yes_no_question(sample_data: np.ndarray, prediction: Any, 
                                   model_info: Dict, question_idx: int) -> Dict[str, Any]:
    """Generate yes/no question"""
    
    # Create question based on model type and prediction
    model_type = model_info['model_type']
    
    if 'classification' in model_type.lower() or model_type in ['logistic', 'svm_classifier', 'random_forest_classifier']:
        question_text = f"Based on the input features {sample_data[:3].round(2).tolist()}..., do you think the prediction '{prediction}' is correct?"
    elif model_type == 'kmeans':
        question_text = f"For data point with features {sample_data[:3].round(2).tolist()}..., do you agree it belongs to cluster {prediction}?"
    else:
        question_text = f"Given the input {sample_data[:3].round(2).tolist()}..., is the predicted value {prediction:.2f} reasonable?"
    
    return {
        'question_id': f"q_{question_idx + 1}",
        'question_text': question_text,
        'question_type': 'yes_no',
        'sample_data': sample_data.tolist(),
        'prediction': float(prediction) if isinstance(prediction, (int, float, np.number)) else str(prediction),
        'correct_answer': None,  # To be filled by domain expert
        'confidence': np.random.uniform(0.7, 0.95),  # Simulated confidence
        'metadata': {
            'model_type': model_type,
            'sample_index': question_idx
        }
    }

async def _generate_multiple_choice_question(sample_data: np.ndarray, prediction: Any,
                                           model_info: Dict, question_idx: int) -> Dict[str, Any]:
    """Generate multiple choice question"""
    
    model_type = model_info['model_type']
    
    # Generate options based on model type
    if 'classification' in model_type.lower() or model_type in ['logistic', 'svm_classifier', 'random_forest_classifier']:
        options = [
            str(prediction),
            f"Alternative_{prediction}_1", 
            f"Alternative_{prediction}_2",
            "None of the above"
        ]
        question_text = f"For input features {sample_data[:3].round(2).tolist()}..., what should the classification be?"
    else:
        pred_val = float(prediction)
        options = [
            f"{pred_val:.2f}",
            f"{pred_val * 0.8:.2f}",
            f"{pred_val * 1.2:.2f}",
            f"{pred_val * 1.5:.2f}"
        ]
        question_text = f"Given input {sample_data[:3].round(2).tolist()}..., what is the most appropriate predicted value?"
    
    np.random.shuffle(options[1:-1])  # Shuffle middle options, keep prediction first and "None" last
    
    return {
        'question_id': f"q_{question_idx + 1}",
        'question_text': question_text,
        'question_type': 'multiple_choice',
        'options': options,
        'sample_data': sample_data.tolist(),
        'prediction': float(prediction) if isinstance(prediction, (int, float, np.number)) else str(prediction),
        'correct_answer': None,
        'confidence': np.random.uniform(0.6, 0.9),
        'metadata': {
            'model_type': model_type,
            'sample_index': question_idx
        }
    }

async def _generate_open_ended_question(sample_data: np.ndarray, prediction: Any,
                                      model_info: Dict, question_idx: int) -> Dict[str, Any]:
    """Generate open-ended question"""
    
    question_text = f"Analyze this prediction: Input features are {sample_data[:5].round(3).tolist()}, and the model predicted {prediction}. What factors might influence this prediction and how confident are you in its accuracy?"
    
    return {
        'question_id': f"q_{question_idx + 1}",
        'question_text': question_text,
        'question_type': 'open_ended',
        'sample_data': sample_data.tolist(),
        'prediction': float(prediction) if isinstance(prediction, (int, float, np.number)) else str(prediction),
        'correct_answer': None,
        'confidence': np.random.uniform(0.5, 0.8),
        'metadata': {
            'model_type': model_info['model_type'],
            'sample_index': question_idx
        }
    }

@router.post("/answer/{questionnaire_id}")
async def submit_questionnaire_answers(
    questionnaire_id: str,
    answers: str = Form(...),  # JSON string of answers
    agent_response: Optional[str] = Form(None),  # Optional agent response
    db: JSONDatabase = Depends(get_db)
):
    """Submit answers to questionnaire"""
    try:
        # Parse answers
        try:
            answer_data = json.loads(answers)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid answers format")
        
        # Get questionnaire
        questionnaires = await db.get_questionnaires()
        questionnaire = None
        
        for q_id, q_data in questionnaires.items():
            if q_data.get('questionnaire_id') == questionnaire_id:
                questionnaire = q_data
                break
        
        if not questionnaire:
            raise HTTPException(status_code=404, detail="Questionnaire not found")
        
        # Update questionnaire with answers
        questionnaire['answers'] = answer_data
        questionnaire['agent_response'] = agent_response
        questionnaire['status'] = 'answered'
        questionnaire['answered_at'] = 'now'
        
        # Process answers and generate feedback
        feedback = await _process_questionnaire_feedback(questionnaire, answer_data)
        questionnaire['feedback'] = feedback
        
        # Update in database
        file_path = os.path.join(db.base_path, db.files['questionnaires'])
        all_questionnaires = await db._read_file(file_path)
        
        for q_id, q_data in all_questionnaires.items():
            if q_data.get('questionnaire_id') == questionnaire_id:
                all_questionnaires[q_id] = questionnaire
                break
        
        await db._write_file(file_path, all_questionnaires)
        
        # Log iteration
        await db.log_iteration({
            'agent_id': questionnaire['agent_id'],
            'type': 'questionnaire_answered',
            'questionnaire_id': questionnaire_id,
            'model_id': questionnaire['model_id'],
            'feedback_score': feedback.get('score', 0),
            'status': 'completed'
        })
        
        return {
            'questionnaire_id': questionnaire_id,
            'status': 'answered',
            'feedback': feedback,
            'total_questions': len(questionnaire['questions']),
            'answered_questions': len(answer_data)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting answers: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def _process_questionnaire_feedback(questionnaire: Dict, answers: Dict) -> Dict[str, Any]:
    """Process questionnaire answers and generate feedback"""
    try:
        total_questions = len(questionnaire['questions'])
        answered_questions = len(answers)
        
        # Calculate basic metrics
        completion_rate = answered_questions / total_questions if total_questions > 0 else 0
        
        # Analyze answers for patterns
        positive_answers = 0
        negative_answers = 0
        
        for question_id, answer in answers.items():
            if isinstance(answer, str):
                if answer.lower() in ['yes', 'correct', 'agree', 'true']:
                    positive_answers += 1
                elif answer.lower() in ['no', 'incorrect', 'disagree', 'false']:
                    negative_answers += 1
        
        # Calculate feedback score (0-1)
        if answered_questions > 0:
            confidence_score = positive_answers / answered_questions
        else:
            confidence_score = 0.5
        
        # Generate feedback insights
        insights = []
        
        if confidence_score > 0.8:
            insights.append("High confidence in model predictions")
        elif confidence_score < 0.3:
            insights.append("Low confidence indicates potential model issues")
            insights.append("Consider retraining with additional data")
        else:
            insights.append("Mixed confidence - model may need refinement")
        
        if completion_rate < 0.8:
            insights.append("Low completion rate - questions may be too complex")
        
        return {
            'score': confidence_score,
            'completion_rate': completion_rate,
            'positive_answers': positive_answers,
            'negative_answers': negative_answers,
            'insights': insights,
            'recommendation': 'retrain' if confidence_score < 0.4 else 'continue'
        }
        
    except Exception as e:
        logger.error(f"Error processing feedback: {e}")
        return {'error': str(e), 'score': 0.0}

@router.get("/list/{agent_id}")
async def list_questionnaires(
    agent_id: str,
    status: Optional[str] = None,
    db: JSONDatabase = Depends(get_db)
):
    """List questionnaires for an agent"""
    try:
        questionnaires = await db.get_questionnaires(agent_id)
        
        if status:
            questionnaires = [q for q in questionnaires if q.get('status') == status]
        
        return {
            'agent_id': agent_id,
            'questionnaires': questionnaires,
            'total_count': len(questionnaires)
        }
        
    except Exception as e:
        logger.error(f"Error listing questionnaires: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/details/{questionnaire_id}")
async def get_questionnaire_details(
    questionnaire_id: str,
    db: JSONDatabase = Depends(get_db)
):
    """Get detailed questionnaire information"""
    try:
        questionnaires = await db.get_questionnaires()
        
        for q_data in questionnaires:
            if q_data.get('questionnaire_id') == questionnaire_id:
                return q_data
        
        raise HTTPException(status_code=404, detail="Questionnaire not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting questionnaire details: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{questionnaire_id}")
async def delete_questionnaire(
    questionnaire_id: str,
    db: JSONDatabase = Depends(get_db)
):
    """Delete a questionnaire"""
    try:
        file_path = os.path.join(db.base_path, db.files['questionnaires'])
        questionnaires = await db._read_file(file_path)
        
        # Find and remove questionnaire
        removed = False
        for q_id, q_data in list(questionnaires.items()):
            if q_data.get('questionnaire_id') == questionnaire_id:
                del questionnaires[q_id]
                removed = True
                break
        
        if not removed:
            raise HTTPException(status_code=404, detail="Questionnaire not found")
        
        # Save updated data
        await db._write_file(file_path, questionnaires)
        
        return {'message': f'Questionnaire {questionnaire_id} deleted successfully'}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting questionnaire: {e}")
        raise HTTPException(status_code=500, detail=str(e))