#!/usr/bin/env python3
"""
웹 브라우저 기반 픽셀 좌표 수집 도구
서버 환경에서 SSH를 통해 접속하여 사용할 수 있습니다.
"""

import os
import cv2
import numpy as np
import json
from pathlib import Path
import argparse
from flask import Flask, render_template_string, request, jsonify
import base64
from io import BytesIO

app = Flask(__name__)

class PixelCoordinateCollector:
    def __init__(self, output_dir="./data/pixel_points"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 현재 처리 중인 변수들
        self.current_video_files = []
        self.current_video_index = 0
        
        # 현재 선택된 guide 이미지
        self.current_guide = "guide1"  # 기본값
        
    def find_video_files(self, video_dir):
        """비디오 파일들을 찾습니다."""
        video_dir = Path(video_dir)
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(video_dir.glob(f'*{ext}'))
            video_files.extend(video_dir.glob(f'*{ext.upper()}'))
        
        self.current_video_files = sorted(video_files)
        self.current_video_index = 0
        
        return len(self.current_video_files)
    
    def get_guide_path(self):
        """현재 선택된 guide 이미지 경로 반환"""
        return f"/home/jhbae/pianomime/data/{self.current_guide}.png"

# 전역 설정 객체
pixel_collector = None

# HTML 템플릿
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🎯 Pixel Coordinate Collector</title>
    <style>
        * { box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            margin: 0; 
            padding: 10px; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            min-height: 100vh;
        }
        .container { 
            width: 100%; 
            margin: 0; 
            background: white; 
            border-radius: 15px; 
            box-shadow: 0 10px 30px rgba(0,0,0,0.2); 
            overflow: hidden;
            display: grid;
            grid-template-rows: auto auto 1fr auto;
            min-height: calc(100vh - 20px);
        }
        .app-bar { 
            background: linear-gradient(45deg, #1e3c72, #2a5298); 
            color: white; 
            padding: 15px 20px; 
            display: flex;
            align-items: center;
            justify-content: space-between;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        }
        
        .app-bar-title {
            font-size: 1.5em;
            font-weight: 600;
            text-shadow: 1px 1px 3px rgba(0,0,0,0.3);
        }
        
        .video-selector {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .video-selector label {
            font-size: 0.9em;
            opacity: 0.9;
        }
        
        .video-dropdown {
            background: rgba(255,255,255,0.2);
            color: white;
            border: 1px solid rgba(255,255,255,0.3);
            border-radius: 6px;
            padding: 8px 12px;
            font-size: 0.9em;
            min-width: 200px;
            cursor: pointer;
        }
        
        .video-dropdown option {
            background: #2a5298;
            color: white;
        }
        
        .video-dropdown:focus {
            outline: none;
            border-color: rgba(255,255,255,0.6);
            box-shadow: 0 0 0 2px rgba(255,255,255,0.2);
        }
        
        .main-content {
            display: grid;
            grid-template-columns: 1fr 350px;
            gap: 20px;
            padding: 20px;
            height: 100%;
        }
        
        .left-panel {
            display: flex;
            flex-direction: column;
            gap: 15px;
        }
        
        .right-panel {
            display: flex;
            flex-direction: column;
            gap: 15px;
        }
        
        .video-info { 
            background: linear-gradient(135deg, #e3f2fd, #bbdefb); 
            padding: 20px; 
            border-radius: 10px; 
            border-left: 5px solid #2196f3;
            font-size: 1.1em;
        }
        
        .image-container { 
            flex: 1;
            display: flex;
            justify-content: center;
            align-items: center;
            background: #f8f9fa;
            border-radius: 10px;
            padding: 10px;
            box-shadow: inset 0 2px 5px rgba(0,0,0,0.1);
            position: relative;
        }
        
        #video-canvas { 
            max-width: 100%; 
            max-height: 100%;
            cursor: crosshair; 
            border: 3px solid #ddd; 
            border-radius: 8px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        
        .coordinate-tooltip {
            position: absolute;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 5px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-family: monospace;
            pointer-events: none;
            z-index: 1000;
            transform: translate(10px, -30px);
            white-space: nowrap;
            display: none;
        }
        
        .coordinate-tooltip::after {
            content: '';
            position: absolute;
            top: 100%;
            left: 8px;
            width: 0;
            height: 0;
            border-left: 4px solid transparent;
            border-right: 4px solid transparent;
            border-top: 4px solid rgba(0, 0, 0, 0.8);
        }
        
        .instructions-panel {
            background: linear-gradient(135deg, #f3e5f5, #e1bee7);
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid #9c27b0;
            margin-bottom: 15px;
        }
        
        .instructions-panel h3 {
            margin: 0 0 15px 0;
            color: #6a1b9a;
            font-size: 1.2em;
        }
        
        .instructions-list {
            margin: 0;
            padding-left: 20px;
            color: #4a148c;
        }
        
        .instructions-list li {
            margin-bottom: 8px;
            line-height: 1.4;
        }
        
        .step-highlight {
            background: rgba(156, 39, 176, 0.1);
            padding: 2px 6px;
            border-radius: 4px;
            font-weight: 600;
        }
        
        .status-panel { 
            background: linear-gradient(135deg, #fff3e0, #ffe0b2); 
            padding: 20px; 
            border-radius: 10px; 
            border-left: 5px solid #ff9800;
        }
        
        .status-panel h3 {
            margin: 0 0 15px 0;
            color: #e65100;
            font-size: 1.2em;
        }
        
        #current-status {
            font-size: 1.1em;
            line-height: 1.4;
            margin-bottom: 15px;
            padding: 10px;
            background: rgba(255,255,255,0.7);
            border-radius: 8px;
        }
        
        .points-list { 
            flex: 1;
            background: #f8f9fa; 
            padding: 15px; 
            border-radius: 10px; 
            font-family: 'Courier New', monospace; 
            font-size: 13px;
            overflow-y: auto;
            border: 2px solid #e9ecef;
            max-height: 300px;
        }
        
        .points-list::-webkit-scrollbar {
            width: 8px;
        }
        
        .points-list::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 4px;
        }
        
        .points-list::-webkit-scrollbar-thumb {
            background: #c1c1c1;
            border-radius: 4px;
        }
        
        .controls { 
            background: #f8f9fa;
            padding: 20px; 
            text-align: center;
            border-top: 1px solid #e9ecef;
        }
        
        .button-group {
            display: inline-flex;
            gap: 10px;
            flex-wrap: wrap;
            justify-content: center;
        }
        
        button { 
            background: linear-gradient(45deg, #2196f3, #21cbf3); 
            color: white; 
            border: none; 
            padding: 12px 24px; 
            border-radius: 25px; 
            cursor: pointer; 
            font-size: 16px;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(33, 150, 243, 0.3);
        }
        
        button:hover { 
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(33, 150, 243, 0.4);
        }
        
        button:disabled { 
            background: #ccc; 
            cursor: not-allowed; 
            transform: none;
            box-shadow: none;
        }
        
        .skip-btn { 
            background: linear-gradient(45deg, #ff9800, #ffc107); 
            box-shadow: 0 4px 15px rgba(255, 152, 0, 0.3);
        }
        
        .skip-btn:hover { 
            box-shadow: 0 6px 20px rgba(255, 152, 0, 0.4);
        }
        
        .reset-btn {
            background: linear-gradient(45deg, #f44336, #ff5722);
            box-shadow: 0 4px 15px rgba(244, 67, 54, 0.3);
        }
        
        .reset-btn:hover {
            box-shadow: 0 6px 20px rgba(244, 67, 54, 0.4);
        }
        
        .save-btn {
            background: linear-gradient(45deg, #4caf50, #8bc34a);
            box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
        }
        
        .save-btn:hover {
            box-shadow: 0 6px 20px rgba(76, 175, 80, 0.4);
        }
        
        @media (max-width: 1024px) {
            .main-content {
                grid-template-columns: 1fr;
                grid-template-rows: auto auto 1fr auto;
            }
            
            .right-panel {
                order: -1;
            }
            
            .image-container {
                min-height: 400px;
            }
        }
        
        @media (max-width: 768px) {
            body { padding: 5px; }
            .container { border-radius: 10px; }
            .main-content { padding: 15px; gap: 15px; }
            .app-bar { flex-direction: column; gap: 10px; text-align: center; }
            .app-bar-title { font-size: 1.3em; }
            .app-bar > div { flex-direction: column; gap: 10px; }
            .video-selector { justify-content: center; }
            .video-dropdown { min-width: 250px; }
            .button-group { flex-direction: column; align-items: center; }
            button { width: 200px; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="app-bar">
            <div class="app-bar-title">🎯 Pixel Coordinate Collector</div>
            <div style="display: flex; gap: 20px; align-items: center;">
                <div class="video-selector">
                    <label for="video-select">📁 비디오 선택:</label>
                    <select id="video-select" class="video-dropdown" onchange="selectVideo()">
                        <option value="">로딩 중...</option>
                    </select>
                </div>
                <div class="video-selector">
                    <label for="guide-select">🎯 가이드 선택:</label>
                    <select id="guide-select" class="video-dropdown" onchange="selectGuide()">
                        <option value="guide1">Guide 1</option>
                        <option value="guide2">Guide 2</option>
                        <option value="guide3">Guide 3</option>
                        <option value="guide4">Guide 4</option>
                        <option value="guide5">Guide 5</option>
                        <option value="guide6">Guide 6</option>
                    </select>
                </div>
            </div>
        </div>
        
        <div class="video-info">
            <div id="video-info">🎬 비디오 로딩 중...</div>
        </div>
        
        <div class="main-content">
            <div class="left-panel">
                <div class="image-container">
                    <canvas id="video-canvas"></canvas>
                    <div id="coordinate-tooltip" class="coordinate-tooltip"></div>
                </div>
            </div>
            
            <div class="right-panel">
                <div class="instructions-panel">
                    <h3>📝 사용 방법</h3>
                    <ol class="instructions-list">
                        <li><span class="step-highlight">🖱️ 클릭</span>: 원하는 위치를 마우스로 클릭하세요</li>
                        <li><span class="step-highlight">📌 추가</span>: 필요한 만큼 점을 추가할 수 있습니다</li>
                        <li><span class="step-highlight">💾 저장</span>: 완료되면 저장 버튼을 클릭하세요</li>
                        <li><span class="step-highlight">🔄 리셋</span>: 다시 시작하려면 리셋 버튼을 클릭하세요</li>
                    </ol>
                    <div style="background: rgba(76, 175, 80, 0.1); padding: 8px; border-radius: 4px; margin-top: 10px; font-size: 0.9em;">
                        💡 <strong>팁:</strong> 마우스를 움직이면 현재 좌표가 표시됩니다
                    </div>
                    <div style="background: rgba(33, 150, 243, 0.1); padding: 8px; border-radius: 4px; margin-top: 8px; font-size: 0.9em;">
                        ⌨️ <strong>단축키:</strong> 
                        <span style="font-family: monospace; background: rgba(255,255,255,0.8); padding: 2px 4px; border-radius: 3px; margin: 0 2px;">Ctrl+Z</span> 되돌리기 | 
                        <span style="font-family: monospace; background: rgba(255,255,255,0.8); padding: 2px 4px; border-radius: 3px; margin: 0 2px;">Ctrl+Y</span> 다시 실행
                    </div>
                </div>
                
                <div class="status-panel">
                    <h3>📊 현재 상태:</h3>
                    <div id="current-status">클릭 대기 중...</div>
                </div>
                
                <div class="points-list" id="points-list">
                    <strong>📌 클릭된 좌표:</strong><br>
                </div>
            </div>
        </div>
        
        <div class="controls">
            <div class="button-group">
                <button onclick="undoLastPoint()" id="undo-btn" title="Ctrl+Z">↶ 되돌리기</button>
                <button onclick="redoLastPoint()" id="redo-btn" title="Ctrl+Y">↷ 다시 실행</button>
                <button onclick="resetPoints()" class="reset-btn">🔄 리셋</button>
                <button onclick="skipVideo()" class="skip-btn">⏭️ 건너뛰기</button>
                <button onclick="savePixelPoints()" id="save-btn" class="save-btn">💾 저장</button>
            </div>
        </div>
    </div>

    <script>
        let pixelPoints = [];
        let undoStack = [];
        let redoStack = [];
        let currentVideoInfo = null;
        let canvas = document.getElementById('video-canvas');
        let ctx = canvas.getContext('2d');
        let videoImage = new Image();

        window.onload = function() {
            loadVideoList();
            loadVideoInfo();
            updateStatus();
            loadGuideSelection();
            
            // 키보드 이벤트 리스너 추가
            document.addEventListener('keydown', function(event) {
                if (event.ctrlKey) {
                    if (event.key === 'z' || event.key === 'Z') {
                        event.preventDefault();
                        undoLastPoint();
                    } else if (event.key === 'y' || event.key === 'Y') {
                        event.preventDefault();
                        redoLastPoint();
                    }
                }
            });
        };

        async function loadVideoInfo() {
            try {
                const response = await fetch('/api/video_info');
                const data = await response.json();
                
                if (data.error) {
                    alert(data.error);
                    return;
                }
                
                currentVideoInfo = data;
                
                document.getElementById('video-info').innerHTML = 
                    `<div style="display: flex; align-items: center; gap: 10px; flex-wrap: wrap;">
                        <div>
                            <strong>🎬 ${data.video_name}</strong>
                        </div>
                        <div style="flex: 1;"></div>
                        <div style="background: rgba(255,255,255,0.8); padding: 5px 10px; border-radius: 15px; font-size: 0.9em;">
                            📊 진행: ${data.index}/${data.total}
                        </div>
                        <div style="padding: 3px 8px; border-radius: 12px; font-size: 0.85em; ${data.is_processed ? 'background: #4caf50; color: white;' : 'background: #ff9800; color: white;'}">
                            ${data.is_processed ? '✅ 처리완료' : '⏳ 대기중'}
                        </div>
                    </div>`;
                
                videoImage.onload = function() {
                    canvas.width = this.width;
                    canvas.height = this.height;
                    drawCanvas();
                };
                videoImage.src = data.image_data;
                
            } catch (error) {
                console.error('Error loading video info:', error);
                alert('비디오 정보 로딩 실패');
            }
        }

        function drawCanvas() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(videoImage, 0, 0);
            
            // 클릭된 포인트들 그리기
            pixelPoints.forEach((point, index) => {
                ctx.fillStyle = '#FF0000';
                ctx.beginPath();
                ctx.arc(point[0], point[1], 5, 0, 2 * Math.PI);
                ctx.fill();
                
                ctx.fillStyle = 'white';
                ctx.font = '12px Arial';
                ctx.strokeStyle = 'black';
                ctx.lineWidth = 2;
                ctx.strokeText(index + 1, point[0] + 8, point[1] - 8);
                ctx.fillText(index + 1, point[0] + 8, point[1] - 8);
            });
        }

        canvas.onclick = function(event) {
            const rect = canvas.getBoundingClientRect();
            const scaleX = canvas.width / rect.width;
            const scaleY = canvas.height / rect.height;
            
            const x = Math.round((event.clientX - rect.left) * scaleX);
            const y = Math.round((event.clientY - rect.top) * scaleY);
            
            // 현재 상태를 undo 스택에 저장
            undoStack.push([...pixelPoints]);
            // 새로운 액션이므로 redo 스택 초기화
            redoStack = [];
            
            pixelPoints.push([x, y]);
            console.log(`클릭 ${pixelPoints.length}: (${x}, ${y})`);
            
            drawCanvas();
            updateUI();
        };

        // 마우스 이동 시 좌표 표시
        canvas.onmousemove = function(event) {
            const tooltip = document.getElementById('coordinate-tooltip');
            const rect = canvas.getBoundingClientRect();
            const scaleX = canvas.width / rect.width;
            const scaleY = canvas.height / rect.height;
            
            const x = Math.round((event.clientX - rect.left) * scaleX);
            const y = Math.round((event.clientY - rect.top) * scaleY);
            
            tooltip.textContent = `(${x}, ${y})`;
            tooltip.style.left = event.clientX + 'px';
            tooltip.style.top = event.clientY + 'px';
            tooltip.style.display = 'block';
        };

        canvas.onmouseleave = function() {
            document.getElementById('coordinate-tooltip').style.display = 'none';
        };

        canvas.onmouseenter = function() {
            document.getElementById('coordinate-tooltip').style.display = 'block';
        };

        function updateUI() {
            updateStatus();
            updatePointsList();
        }

        function updateStatus() {
            const statusElement = document.getElementById('current-status');
            
            // Undo/Redo 상태 정보
            const undoAvailable = undoStack.length > 0;
            const redoAvailable = redoStack.length > 0;
            const undoText = undoAvailable ? '✅' : '❌';
            const redoText = redoAvailable ? '✅' : '❌';
            
            // 버튼 상태 업데이트
            const undoBtn = document.getElementById('undo-btn');
            const redoBtn = document.getElementById('redo-btn');
            if (undoBtn) undoBtn.disabled = !undoAvailable;
            if (redoBtn) redoBtn.disabled = !redoAvailable;
            
            if (pixelPoints.length === 0) {
                statusElement.innerHTML = `
                    <div style="background: rgba(255,255,255,0.9); padding: 8px; border-radius: 6px;">
                        🖱️ <strong>클릭 대기 중</strong><br>
                        <span style="font-size: 0.9em; color: #666;">원하는 위치를 클릭하세요</span>
                        <div style="margin-top: 8px; font-size: 0.8em; color: #888; border-top: 1px solid #eee; padding-top: 5px;">
                            ${undoText} Ctrl+Z 되돌리기 | ${redoText} Ctrl+Y 다시 실행
                        </div>
                    </div>
                `;
            } else {
                statusElement.innerHTML = `
                    <div style="background: rgba(76, 175, 80, 0.1); padding: 8px; border-radius: 6px;">
                        📌 <strong>${pixelPoints.length}개 좌표 수집됨</strong><br>
                        <span style="font-size: 0.9em; color: #333;">더 추가하거나 저장하세요</span>
                        <div style="margin-top: 8px; font-size: 0.8em; color: #666; border-top: 1px solid #ddd; padding-top: 5px;">
                            ${undoText} Ctrl+Z 되돌리기 | ${redoText} Ctrl+Y 다시 실행
                        </div>
                    </div>
                `;
            }
        }

        function updatePointsList() {
            const listElement = document.getElementById('points-list');
            let content = '<strong>📌 클릭된 좌표:</strong><br><br>';
            
            pixelPoints.forEach((point, index) => {
                content += `🔴 ${index + 1}번: (${point[0]}, ${point[1]})<br>`;
            });
            
            if (pixelPoints.length === 0) {
                content += '<span style="color: #666; font-style: italic;">아직 클릭된 좌표가 없습니다.</span>';
            }
            
            listElement.innerHTML = content;
            listElement.scrollTop = listElement.scrollHeight;
        }

        function undoLastPoint() {
            if (undoStack.length > 0) {
                // 현재 상태를 redo 스택에 저장
                redoStack.push([...pixelPoints]);
                // 이전 상태로 복원
                pixelPoints = undoStack.pop();
                
                console.log(`Undo: ${pixelPoints.length}개 좌표로 복원`);
                drawCanvas();
                updateUI();
            } else {
                console.log('Undo: 되돌릴 액션이 없습니다');
            }
        }

        function redoLastPoint() {
            if (redoStack.length > 0) {
                // 현재 상태를 undo 스택에 저장
                undoStack.push([...pixelPoints]);
                // 다시 실행 상태로 복원
                pixelPoints = redoStack.pop();
                
                console.log(`Redo: ${pixelPoints.length}개 좌표로 복원`);
                drawCanvas();
                updateUI();
            } else {
                console.log('Redo: 다시 실행할 액션이 없습니다');
            }
        }

        function resetPoints() {
            console.log('Reset points called');
            // 리셋도 되돌릴 수 있도록 현재 상태를 undo 스택에 저장
            if (pixelPoints.length > 0) {
                undoStack.push([...pixelPoints]);
                redoStack = [];
            }
            pixelPoints = [];
            drawCanvas();
            updateUI();
        }

        async function savePixelPoints() {
            if (pixelPoints.length === 0) {
                alert('최소 1개 이상의 좌표를 클릭하세요.');
                return;
            }
            
            try {
                const response = await fetch('/api/save_pixel_points', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        pixel_points: pixelPoints
                    })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    alert(`✅ ${pixelPoints.length}개 좌표 저장 완료!`);
                    loadVideoList();
                    nextVideo();
                } else {
                    alert('❌ 저장 실패: ' + data.error);
                }
                
            } catch (error) {
                console.error('Error saving pixel points:', error);
                alert('저장 중 오류 발생');
            }
        }

        async function skipVideo() {
            if (confirm('현재 비디오를 건너뛰시겠습니까?')) {
                pixelPoints = [];
                undoStack = [];
                redoStack = [];
                await fetch('/api/skip_video');
                loadVideoList();
                nextVideo();
            }
        }

        async function nextVideo() {
            pixelPoints = [];
            undoStack = [];
            redoStack = [];
            await fetch('/api/next_video');
            loadVideoList();
            await loadVideoInfo();
            updateUI();
        }

        async function loadVideoList() {
            try {
                const response = await fetch('/api/video_list');
                const data = await response.json();
                
                const dropdown = document.getElementById('video-select');
                dropdown.innerHTML = '';
                
                data.videos.forEach((video, index) => {
                    const option = document.createElement('option');
                    option.value = video.index;
                    const statusIcon = video.is_processed ? '✅' : '⏳';
                    option.textContent = `${statusIcon} ${video.name}`;
                    
                    if (video.index === data.current_index) {
                        option.selected = true;
                    }
                    
                    dropdown.appendChild(option);
                });
                
            } catch (error) {
                console.error('비디오 목록 로딩 실패:', error);
            }
        }

        async function selectVideo() {
            const dropdown = document.getElementById('video-select');
            const selectedIndex = dropdown.value;
            
            if (selectedIndex === '') return;
            
            try {
                pixelPoints = [];
                undoStack = [];
                redoStack = [];
                const response = await fetch(`/api/select_video/${selectedIndex}`);
                const data = await response.json();
                
                if (data.success) {
                    await loadVideoInfo();
                    updateUI();
                } else {
                    alert('비디오 선택 실패: ' + data.error);
                }
                
            } catch (error) {
                console.error('비디오 선택 실패:', error);
                alert('비디오 선택 중 오류 발생');
            }
        }

        async function selectGuide() {
            const dropdown = document.getElementById('guide-select');
            const selectedGuide = dropdown.value;
            
            if (selectedGuide === '') return;
            
            try {
                const response = await fetch(`/api/select_guide/${selectedGuide}`, {
                    method: 'POST'
                });
                const data = await response.json();
                
                if (data.success) {
                    await loadVideoInfo();
                    updateUI();
                } else {
                    alert('가이드 선택 실패: ' + data.error);
                }
                
            } catch (error) {
                console.error('가이드 선택 실패:', error);
                alert('가이드 선택 중 오류 발생');
            }
        }

        async function loadGuideSelection() {
            try {
                const response = await fetch('/api/current_guide');
                const data = await response.json();
                
                if (data.success) {
                    const dropdown = document.getElementById('guide-select');
                    dropdown.value = data.current_guide;
                }
            } catch (error) {
                console.error('가이드 선택 로딩 실패:', error);
            }
        }
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    """메인 페이지"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/video_info')
def get_video_info():
    """현재 비디오 정보 반환"""
    if pixel_collector.current_video_index >= len(pixel_collector.current_video_files):
        return jsonify({'error': '모든 비디오 처리 완료!'})
    
    video_path = pixel_collector.current_video_files[pixel_collector.current_video_index]
    video_name = video_path.stem
    
    # 이미 처리된 파일인지 확인
    pixel_file = pixel_collector.output_dir / f"{video_name}_pixel_points.json"
    is_processed = pixel_file.exists()
    
    # 비디오 중간 프레임을 base64로 인코딩
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return jsonify({'error': '비디오 파일을 열 수 없습니다.'})
    
    # 전체 프레임 수 구하기
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 중간 프레임 위치로 이동
    middle_frame = total_frames // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return jsonify({'error': '비디오 프레임을 읽을 수 없습니다.'})
    
    # 가이드 이미지 오버레이
    guide_path = pixel_collector.get_guide_path()
    if os.path.exists(guide_path):
        try:
            # 가이드 이미지 로드
            guide_img = cv2.imread(guide_path, cv2.IMREAD_UNCHANGED)
            
            if guide_img is not None:
                # 가이드 이미지를 비디오 프레임 크기에 맞게 리사이즈
                frame_height, frame_width = frame.shape[:2]
                guide_resized = cv2.resize(guide_img, (frame_width, frame_height))
                
                # 알파 채널이 있는 경우 (PNG 투명도 처리)
                if guide_resized.shape[2] == 4:
                    # 알파 채널 분리
                    alpha_channel = guide_resized[:, :, 3] / 255.0
                    guide_rgb = guide_resized[:, :, :3]
                    
                    # 알파 블렌딩
                    for c in range(3):
                        frame[:, :, c] = frame[:, :, c] * (1 - alpha_channel) + guide_rgb[:, :, c] * alpha_channel
                else:
                    # 알파 채널이 없는 경우 50% 투명도로 오버레이
                    frame = cv2.addWeighted(frame, 0.7, guide_resized, 0.3, 0)
                
                print(f"✅ 가이드 이미지 오버레이 완료: {guide_path}")
        except Exception as e:
            print(f"⚠️ 가이드 이미지 오버레이 실패: {e}")
    else:
        print(f"⚠️ 가이드 이미지를 찾을 수 없음: {guide_path}")
    
    # base64 인코딩
    _, buffer = cv2.imencode('.jpg', frame)
    img_str = base64.b64encode(buffer).decode()
    
    return jsonify({
        'video_path': str(video_path),
        'video_name': video_name,
        'index': pixel_collector.current_video_index + 1,
        'total': len(pixel_collector.current_video_files),
        'is_processed': is_processed,
        'image_data': f"data:image/jpeg;base64,{img_str}"
    })

@app.route('/api/save_pixel_points', methods=['POST'])
def save_pixel_points():
    """픽셀 좌표 데이터 저장"""
    data = request.get_json()
    
    if not data or 'pixel_points' not in data or len(data['pixel_points']) == 0:
        return jsonify({'error': '최소 1개 이상의 좌표가 필요합니다.'})
    
    video_path = pixel_collector.current_video_files[pixel_collector.current_video_index]
    video_name = video_path.stem
    
    try:
        # 결과 저장
        pixel_file_json = pixel_collector.output_dir / f"{video_name}_pixel_points.json"
        pixel_file_npy = pixel_collector.output_dir / f"{video_name}_pixel_points.npy"
        
        # JSON 형태로 저장 (가독성)
        save_data = {
            'video_name': video_name,
            'video_path': str(video_path),
            'pixel_points': data['pixel_points'],
            'point_count': len(data['pixel_points']),
            'description': '웹 도구로 수집된 픽셀 좌표'
        }
        
        with open(pixel_file_json, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        # NumPy 형태로도 저장 (data_preprocessing.py와 호환)
        pixel_points_array = np.array(data['pixel_points'], dtype='float32')
        np.save(pixel_file_npy, pixel_points_array)
        
        return jsonify({
            'success': True, 
            'message': f'{len(data["pixel_points"])}개 좌표 저장 완료',
            'saved_files': [str(pixel_file_json), str(pixel_file_npy)]
        })
        
    except Exception as e:
        return jsonify({'error': f'저장 중 오류: {e}'})

@app.route('/api/next_video')
def next_video():
    """다음 비디오로 이동"""
    pixel_collector.current_video_index += 1
    return jsonify({'success': True})

@app.route('/api/skip_video')
def skip_video():
    """현재 비디오 건너뛰기"""
    pixel_collector.current_video_index += 1
    return jsonify({'success': True})

@app.route('/api/video_list')
def get_video_list():
    """비디오 파일 목록 반환"""
    video_list = []
    for i, video_path in enumerate(pixel_collector.current_video_files):
        video_name = video_path.stem
        pixel_file = pixel_collector.output_dir / f"{video_name}_pixel_points.json"
        is_processed = pixel_file.exists()
        
        video_list.append({
            'index': i,
            'name': video_name,
            'path': str(video_path),
            'is_processed': is_processed
        })
    
    return jsonify({
        'videos': video_list,
        'current_index': pixel_collector.current_video_index,
        'total': len(pixel_collector.current_video_files)
    })

@app.route('/api/select_video/<int:video_index>')
def select_video(video_index):
    """특정 비디오 선택"""
    if 0 <= video_index < len(pixel_collector.current_video_files):
        pixel_collector.current_video_index = video_index
        return jsonify({'success': True, 'message': f'비디오 {video_index + 1} 선택됨'})
    else:
        return jsonify({'error': '유효하지 않은 비디오 인덱스'})

@app.route('/api/select_guide/<string:guide_name>', methods=['POST'])
def select_guide(guide_name):
    """특정 가이드 선택"""
    if guide_name in ['guide1', 'guide2', 'guide3', 'guide4', 'guide5', 'guide6']:
        pixel_collector.current_guide = guide_name
        return jsonify({'success': True, 'message': f'{guide_name} 선택됨'})
    else:
        return jsonify({'error': '유효하지 않은 가이드 이름'})

@app.route('/api/current_guide')
def get_current_guide():
    """현재 선택된 가이드 반환"""
    return jsonify({'success': True, 'current_guide': pixel_collector.current_guide})

def main():
    parser = argparse.ArgumentParser(description='웹 기반 픽셀 좌표 수집 도구')
    parser.add_argument('--video_dir', type=str, default='./data/video',
                       help='비디오 파일이 있는 디렉토리')
    parser.add_argument('--output_dir', type=str, default='./data/pixel_points',
                       help='픽셀 좌표 결과 저장 디렉토리')  
    parser.add_argument('--port', type=int, default=5001,
                       help='웹 서버 포트 (기본값: 5001)')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='웹 서버 호스트 (기본값: 0.0.0.0)')
    
    args = parser.parse_args()
    
    print("🎯 Pixel Coordinate Collector")
    print("=" * 50)
    print(f"📁 비디오 디렉토리: {args.video_dir}")
    print(f"💾 출력 디렉토리: {args.output_dir}")
    print(f"🌐 서버: http://localhost:{args.port}")
    print("=" * 50)
    
    # 설정 초기화
    global pixel_collector
    pixel_collector = PixelCoordinateCollector(output_dir=args.output_dir)
    
    # 비디오 파일 찾기
    video_count = pixel_collector.find_video_files(args.video_dir)
    print(f"🎬 총 {video_count}개의 비디오 파일 발견")
    
    if video_count == 0:
        print("❌ 비디오 파일을 찾을 수 없습니다.")
        return
    
    print(f"\n🚀 웹 서버 시작...")
    print(f"🌐 브라우저에서 http://localhost:{args.port} 접속")
    print(f"🔗 SSH 포트포워딩: ssh -L {args.port}:localhost:{args.port} user@server")
    
    # Flask 앱 실행
    app.run(host=args.host, port=args.port, debug=False)

if __name__ == "__main__":
    main() 