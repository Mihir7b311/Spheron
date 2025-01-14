import React, { useState } from 'react';
import { ChevronDown, ChevronUp, Calendar, Clock, Check, Settings2 } from 'lucide-react';
import '../styles/SchedulingSystem.css';

const SchedulingSystem = ({ onScheduleChange }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [scheduleConfig, setScheduleConfig] = useState({
    period: 'daily',
    startDate: '',
    time: '',
    selectedDays: [],
    customInterval: ''
  });

  const periods = ['daily', 'weekly', 'monthly', 'custom'];
  const daysOfWeek = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];

  const generateScheduleData = () => {
    const scheduleData = {
      type: scheduleConfig.period,
      configuration: {
        startDate: scheduleConfig.startDate,
        time: scheduleConfig.time,
        days: scheduleConfig.selectedDays,
        customInterval: scheduleConfig.customInterval || null
      }
    };

    let readableFormat = `Schedule: ${scheduleConfig.period}`;
    if (scheduleConfig.period === 'weekly' && scheduleConfig.selectedDays.length > 0) {
      readableFormat += ` on ${scheduleConfig.selectedDays.join(', ')}`;
    }
    if (scheduleConfig.period === 'custom' && scheduleConfig.customInterval) {
      readableFormat += `: ${scheduleConfig.customInterval}`;
    }
    if (scheduleConfig.startDate) readableFormat += ` from ${scheduleConfig.startDate}`;
    if (scheduleConfig.time) readableFormat += ` at ${scheduleConfig.time}`;

    return { data: scheduleData, displayText: readableFormat };
  };

  const handleSave = () => {
    const schedule = generateScheduleData();
    onScheduleChange(schedule);
    setIsOpen(false);
  };

  const handleConfigChange = (key, value) => {
    setScheduleConfig((prev) => {
      const updatedConfig = { ...prev, [key]: value };
  
      // Reset fields based on the selected period
      if (key === 'period') {
        if (value === 'daily') {
          updatedConfig.selectedDays = []; // Clear selected days for daily
          updatedConfig.customInterval = ''; // Clear custom interval
        } else if (value === 'weekly') {
          updatedConfig.customInterval = ''; // Clear custom interval for weekly
        } else if (value === 'custom') {
          updatedConfig.selectedDays = []; // Clear selected days for custom
        }
      }
  
      return updatedConfig;
    });
  };
  

  const handleDayToggle = (day) => {
    const updatedDays = scheduleConfig.selectedDays.includes(day)
      ? scheduleConfig.selectedDays.filter(d => d !== day)
      : [...scheduleConfig.selectedDays, day];
    handleConfigChange('selectedDays', updatedDays);
  };

  return (
    <div className="scheduler">
      <button className="scheduler-header" onClick={() => setIsOpen(!isOpen)}>
        <div className="scheduler-title">
          <div className="scheduler-icon">
            <Settings2 size={20} />
          </div>
          <span>Schedule Configuration</span>
        </div>
        {isOpen ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
      </button>
      
      {isOpen && (
        <div className="scheduler-content">
          {/* Period Selection */}
          <div className="input-group">
            <label className="input-label">Schedule Type</label>
            <div className="period-buttons">
              {periods.map(period => (
                <button
                  key={period}
                  onClick={() => handleConfigChange('period', period)}
                  className={`period-button ${scheduleConfig.period === period ? 'active' : ''}`}
                >
                  {period}
                </button>
              ))}
            </div>
          </div>

          {/* Custom Interval */}
          {scheduleConfig.period === 'custom' && (
            <div className="input-group">
              <label className="input-label">Interval</label>
              <input
                type="text"
                className="scheduler-input"
                value={scheduleConfig.customInterval}
                onChange={(e) => handleConfigChange('customInterval', e.target.value)}
                placeholder="e.g., every 3 days"
              />
            </div>
          )}

          {/* Date & Time Selection */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
            <div className="input-group">
              <label className="input-label">Start Date</label>
              <input
                type="date"
                className="scheduler-input"
                value={scheduleConfig.startDate}
                onChange={(e) => handleConfigChange('startDate', e.target.value)}
              />
            </div>

            <div className="input-group">
              <label className="input-label">Time</label>
              <input
                type="time"
                className="scheduler-input"
                value={scheduleConfig.time}
                onChange={(e) => handleConfigChange('time', e.target.value)}
              />
            </div>
          </div>

          {/* Weekly Days Selection */}
          {scheduleConfig.period === 'weekly' && (
            <div className="input-group">
              <label className="input-label">Days of Week</label>
              <div className="days-grid">
                {daysOfWeek.map(day => (
                  <button
                    key={day}
                    onClick={() => handleDayToggle(day)}
                    className={`day-button ${scheduleConfig.selectedDays.includes(day) ? 'active' : ''}`}
                  >
                    {day}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Save Button */}
          <button className="save-button" onClick={handleSave}>
            <Check size={20} />
            <span>Apply Schedule</span>
          </button>
        </div>
      )}
    </div>
  );
};

export default SchedulingSystem;